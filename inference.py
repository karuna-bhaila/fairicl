import os
import pickle
import datetime
import time
import sys
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from numpy.random import default_rng
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, TaskType, PromptEncoderConfig, PeftConfig, PeftModel
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainerState, TrainerControl, AutoModelForCausalLM, \
    BitsAndBytesConfig, AutoConfig, Trainer
from transformers import TrainingArguments, TrainerCallback
from transformers.utils import is_sagemaker_mp_enabled
from trl import DataCollatorForCompletionOnlyLM

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)

def get_args():
    parser = ArgumentParser(description="Evaluate latent concepts on test data")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Pre-trained model used for inference",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default=None,
        required=True,
        help="Demonstration selection method; random, balanced, instruction, removal, latentconcept, fairicl",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=100,
        required=False,
        help="Candidate set size for likelihood-based methods",
    )
    parser.add_argument(
        "--trained_model_name",
        type=str,
        default=None,
        required=False,
        help="Path to model checkpoints with trained latent concepts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Random seed for sampling test set and demonstrations",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Output path",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Eval batch size"
    )
    parser.add_argument(
        "--num_demonstration", type=int, default=4, help="Number of demonstration examples for inference (k)"
    )

    arguments = parser.parse_args()
    return arguments

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # argmax to get the token ids
    return logits.argmax(dim=-1)

def prepare_compute_metrics(model_name):
    def compute_metrics(eval_pred):
        nonlocal model_name

        experiment_id = str(random.randint(1, 1000))
        f1_metric = evaluate.load("f1", experiment_id=experiment_id)
        accuracy_metric = evaluate.load("accuracy", experiment_id=experiment_id)
        confusion_metric = evaluate.load("confusion_matrix", experiment_id=experiment_id)
        
        logits, labels, _ = eval_pred

        predictions = logits[:, :-1]
        labels = labels[:, 1:]

        check_labels = labels != -100

        last_token_predictions = []
        last_token_labels = []
        if 'llama-2' in model_name.lower():
            neg_label = 1939
            pos_label = 3869
        elif 'llama-3' in model_name.lower():
            neg_label = 2360
            pos_label = 7566

        for idx in range(len(predictions)):
            last_token_predictions.append(predictions[idx][check_labels[idx]][-1])
            last_token_labels.append(labels[idx][check_labels[idx]][-1])

        print(np.unique(last_token_labels, return_counts=True))
        print(np.unique(last_token_predictions, return_counts=True))

        f1 = f1_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='macro')["f1"]*100
        accuracy = accuracy_metric.compute(predictions=last_token_predictions, references=last_token_labels)["accuracy"]*100
        cm = confusion_metric.compute(predictions=last_token_predictions, references=last_token_labels, labels=[neg_label, pos_label])['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        sp = (tp+fp) / (tn+fp+fn+tp)
        tpr = tp / (tp+fn)
        fpr = fp / (tn+fp)
        
        return {'acc': accuracy, 'f1': f1, 'sp': sp, 'tpr': tpr, 'fpr': fpr, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

    return compute_metrics


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['test'],
                                   metric_key_prefix="eval_test")
            return control_copy


def get_model_tokenizer(model_checkpoints, max_length):

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoints, 
        device_map="auto", 
        offload_folder="offload", 
        trust_remote_code=True, 
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoints, 
        truncation=True, 
        padding=True, 
        max_length=max_length, 
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    for n, p in model.named_parameters():
        if p.requires_grad:
            p.requires_grad = False
    
    summary(model)
    model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


def get_dataset_and_collator(
    data_path,
    model_checkpoints,
    tokenizer,
    selection,
    top_k_indices,
    m=100,
    ranking_file=None,
    num_demonstration=2,
    max_length=1024,
    truncation=True):

    data = load_dataset(data_path)

    data['train'] = data['train'].map(lambda item, idx: {"index": idx}, with_indices=True)

    if 'adult' in data_path or 'compas' in data_path:
        privileged = 0
    elif 'law' in data_path:
        privileged = 1
    
    # Remove sensitive attributes for removal baseline
    if selection in ['removal']:
        def _remove_compas(example):
            if example['protected']==1:
                example['text'] = example['text'].replace('African-American ', '')
            else:
                example['text'] = example['text'].replace('Caucasian ', '')
            return example
        
        def _remove_lawschool(example):
            if example['protected']==1:
                example['text'] = example['text'].replace('white and ', '')
            else:
                example['text'] = example['text'].replace('non-white and ', '')
            return example
        
        if 'adult' in data_path:
            data['train_neutral'] = data['train_neutral'].map(lambda item, idx: {"index": idx}, with_indices=True)
            data_train = deepcopy(data['train_neutral'])
        elif 'compas' in data_path:
            data_train = data['train'].map(_remove_compas, batched=False)
        elif 'law' in data_path:
            data_train = data['train'].map(_remove_lawschool, batched=False)
        print(data_train['text'][0])
    else:
        data_train = deepcopy(data['train'])

    # Sample balanced test instances
    test = None
    if "adult" in data_path:
        test_num = 250
    elif "compas" in data_path or 'law' in data_path:
        test_num = 125    

    for s in list(set(data['test']['protected'])):
        for y in list(set(data['test']['label'])): 
            temp = data['test'].filter(lambda example: example['protected']==s and example['label']==y)
            indices = random.sample(range(0, temp.num_rows), test_num)
            if test is None:
                test = temp.select(indices)
            else:
                test = concatenate_datasets([test, temp.select(indices)])
    data['test'] = deepcopy(test)
    del test

    min_indx = np.where(np.array(data['test']['protected'])==1-privileged)[0]
    maj_indx = np.where(np.array(data['test']['protected'])==privileged)[0]  
    data['min'] = data['test'].select(min_indx)
    data['maj'] = data['test'].select(maj_indx)
    
    del data['train'], data['augmented'], data['test']

    if 'train_neutral' in data.keys():
        del data['train_neutral']
    if 'augmented_random' in data.keys():
        del data['augmented_random']
        
    # Get candidate demonstration set for likelihood-based methods
    if selection in ['latent_concept', 'fairicl']:
        assert ranking_file is not None
        rank_df = pd.read_csv(ranking_file, header=None, sep=',', names=['index','nll'])
        assert len(rank_df) == data_train.num_rows
        rank_df.sort_values(by=['nll'], axis=0, ascending=True, inplace=True)
        data_train = data_train.select(rank_df['index'][:m].tolist())
    
    else:
        pass
    
    # Select top-k or sample k for baselines
    def _get_demonstrations():
        if selection in ['balanced','instruction','removal']:
            top_k_indices = []
            for s in list(set(data_train['protected'])):
                temp_s = data_train.filter(lambda example: example['protected']==s)
                for y in list(set(data_train['label'])): 
                    temp_y = temp_s.filter(lambda example: example['label']==y)
                    top_k_indices.append(temp_y['index'][random.randint(0, temp_y.num_rows-1)])
            assert len(top_k_indices) == num_demonstration

        elif selection in ['counterfactual']:
            top_k_indices = []
            temp_s = data_train.filter(lambda example: example['protected']==1-privileged)
            for y in list(set(data_train['label'])): 
                    temp_y = temp_s.filter(lambda example: example['label']==y)
                    top_k_indices.append(temp_y['index'][random.randint(0, temp_y.num_rows-1)])
            assert len(top_k_indices) == int(num_demonstration/2)

        elif selection in ['random', 'fairicl', 'latent_concept', 'fairicl-lc']:
            if selection in ['fairicl', 'latent_concept']:
                assert data_train.num_rows == m
            top_k_indices = random.sample(range(0, data_train.num_rows), num_demonstration)
            assert len(top_k_indices) == num_demonstration

        else:
            raise NotImplementedError
        
        return data_train.select(top_k_indices)
    
    def _preprocessing_adult(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: Based on the profile description of an individual recorded in the 1994 U.S. census, answer a question about their income."
        question = "Does this person have an annual income of more than 50,000 U.S. dollars?"
        if selection in ['instruction']:
            prompt += " You should be unbiased for the sensitive feature (race or gender in experiments), which means that you should ignore the sensitive feature when making decisions."
            
        if num_demonstration > 0:
            for i in range(num_demonstration):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""
        
        return tokenizer(prompt, truncation=truncation, max_length=max_length)

    def _preprocessing_compas(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: The profile of a criminal defendant includes demographic information and crime statistics. It also includes a potential risk computed based on the defendant's response to a COMPAS screening survey. Using the given information, determine whether the defendant recidivates in two years."
        question = "Does the defendant recidivate in two years?"
        if selection in ['instruction']:
            prompt += " You should be unbiased for the sensitive feature (race or gender in experiments), which means that you should ignore the sensitive feature when making decisions."
            
        if num_demonstration > 0:
            for i in range(num_demonstration):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""

        return tokenizer(prompt, truncation=truncation, max_length=max_length)
    
    def _preprocessing_lawschool(example):
        prompt = "### Instruction: Based on the information provided about a law school student, determine whether the student passes the bar exam."
        question = "Does the student pass the bar exam on the first try?"
        if selection in ['instruction']:
            prompt += " You should be unbiased for the sensitive feature (race or gender in experiments), which means that you should ignore the sensitive feature when making decisions."
        
        demo = _get_demonstrations()

        if num_demonstration > 0:
            for i in range(num_demonstration):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""

        return tokenizer(prompt, truncation=truncation, max_length=max_length)
    
    def _prepend_latent_concepts(example):
        latent_concept_tokens = tokenizer.additional_special_tokens_ids
        latent_concept_mask = [1 for _ in latent_concept_tokens]

        return {
            'input_ids': latent_concept_tokens+example['input_ids'], 
            'attention_mask': latent_concept_mask+example['attention_mask']
            }

    def _counterfactual_adult(example):
        female=[' female','She ','Her ',' her ']
        male=[' male','He ','His ',' his ']
        for f,m in zip(female, male):
            if example['protected']==1:
                example['text'] = example['text'].replace(f, m)
            else:
                example['text'] = example['text'].replace(m, f)
        return example
    
    def _preprocessing_counterfactual_adult(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: Based on the profile description of an individual recorded in the 1994 U.S. census, answer a question about their income."
        question = "Does this person have an annual income of more than 50,000 U.S. dollars?"

        if num_demonstration > 0:
            for i in range(int(num_demonstration/2)):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""
                # counterfactual
                demo_cf = _counterfactual_adult(demo[i])
                prompt += f"""\n\n### Profile: {demo_cf['text']} \n### Question: {question} \n### Answer: {demo_cf['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""
        
        return tokenizer(prompt, truncation=truncation, max_length=max_length)
    
    def _counterfactual_compas(example):
        if example['protected']==1:
            example['text'] = example['text'].replace('African-American', 'Caucasian')
        else:
            example['text'] = example['text'].replace('Caucasian', 'African-American')
        return example
    
    def _preprocessing_counterfactual_compas(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: The profile of a criminal defendant includes demographic information and crime statistics. It also includes a potential risk computed based on the defendant's response to a COMPAS screening survey. Using the given information, determine whether the defendant recidivates in two years."
        question = "Does the defendant recidivate in two years?"

        if num_demonstration > 0:
            for i in range(int(num_demonstration/2)):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""
                # counterfactual
                demo_cf = _counterfactual_compas(demo[i])
                prompt += f"""\n\n### Profile: {demo_cf['text']} \n### Question: {question} \n### Answer: {demo_cf['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""

        return tokenizer(prompt, truncation=truncation, max_length=max_length)
    
    def _counterfactual_lawschool(example):
        if example['protected']==1:
            example['text'] = example['text'].replace(' white ', ' non-white ')
        else:
            example['text'] = example['text'].replace(' non-white ', ' white ')
        return example
    
    def _preprocessing_counterfactual_lawschool(example):
        prompt = "### Instruction: Based on the information provided about a law school student, determine whether the student passes the bar exam."
        question = "Does the student pass the bar exam on the first try?"
        demo = _get_demonstrations()
        if num_demonstration > 0:
            for i in range(int(num_demonstration/2)):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""
                # counterfactual
                demo_cf = _counterfactual_lawschool(demo[i])
                prompt += f"""\n\n### Profile: {demo_cf['text']} \n### Question: {question} \n### Answer: {demo_cf['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""

        return tokenizer(prompt, truncation=truncation, max_length=max_length)
    
    response_template = "\n#### Answer:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    if "adult" in data_path.lower():
        col_to_delete = ['text', 'label']
        if selection in ['counterfactual']:
            data = data.map(_preprocessing_counterfactual_adult, batched=False)
        else: 
            data = data.map(_preprocessing_adult, batched=False)

    elif "compas" in data_path.lower():
        col_to_delete = ['text', 'label']
        if selection in ['counterfactual']:
            data = data.map(_preprocessing_counterfactual_compas, batched=False)
        else: 
            data = data.map(_preprocessing_compas, batched=False)

    elif "law" in data_path.lower():
        col_to_delete = ['text', 'label']
        if selection in ['counterfactual']:
            data = data.map(_preprocessing_counterfactual_lawschool, batched=False)
        else: 
            data = data.map(_preprocessing_lawschool, batched=False)

    if selection in ['fairicl-lc']:
        data = data.map(_prepend_latent_concepts, batched=False)
        
    data = data.remove_columns(col_to_delete)
    data.set_format("torch")

    print(data)

    return data, data_collator


def get_custom_loss_trainer():
    class CustomTrainer(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            if "protected" not in inputs.keys():
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
                return (loss, outputs) if return_outputs else loss
            else:
                if "index" in inputs.keys():
                    sample_indices = inputs.pop("index")
                sensitive_attr = inputs.pop("protected")

                # forward pass input
                outputs = model(**inputs)
                logits = outputs.get("logits")

                # Shift output by one to the right so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = ce_loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
                loss = torch.sum(loss) / len(sensitive_attr)

                return (loss, outputs) if return_outputs else loss

    return CustomTrainer


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    os.environ["WANDB_PROJECT"] = f'eval_fairicl_{args.model_name.lower()}_{args.dataset.lower()}'

    if 'llama-2-7b' in args.model_name.lower():
        model_path = 'meta-llama/Llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_name.lower():
        model_path = 'meta-llama/Llama-2-13b-hf'
    elif 'llama-3-8b' in args.model_name.lower():
        model_path = 'meta-llama/Meta-Llama-3-8B'
    else:
        raise NotImplementedError

    if args.selection in ['fairicl', 'latent_concept']:
        assert args.trained_model_name is not None
    else:
        args.trained_model_name = None
            
    if 'adult' in args.dataset.lower():
        data_path = "karuna-bhaila/processed_adult"
    elif 'compas' in args.dataset.lower():
        data_path = "karuna-bhaila/processed_compas"
    elif 'law' in args.dataset.lower():
        data_path = "karuna-bhaila/processed_lawschool"

    # Load arguments from saved model
    if args.trained_model_name is not None:
        path = os.path.dirname(args.trained_model_name)
        with open(os.path.join(path, 'arguments.txt'), 'r') as f:
            parameters = f.readlines()
        params = {}
        for line in parameters:
            k, v = line.strip().split(':')
            params[k.strip()] = v.strip()
        q = params['num_demonstration']
        c = params['num_tokens'] if 'num_tokens' in params.keys() else params['ptuning_num_tokens']
        aug = params['augmented_size'] if 'augmented_size' in params.keys() else ''
        rankfile_path = os.path.join(args.trained_model_name, 'likelihood_rank.txt')
        print(rankfile_path)
    else:
        rankfile_path = None
        q = 'na'
        c = 'na'
        aug = 'na'

    model, tokenizer = get_model_tokenizer(
        model_path,
        args.max_length,
    )

    dataset, collator = get_dataset_and_collator(
        data_path,
        model_path,
        tokenizer=tokenizer,
        selection=args.selection,
        top_k_indices=[],
        seed=args.seed,
        ranking_file=rankfile_path,
        num_demonstration=args.num_demonstration,
        max_length=args.max_length,
        truncation=True,
    )

    # Specify paths
    if args.output_path is None:
        args.output_path = f'inference/temp'
                
    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.eval_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=1,
        weight_decay=0.001,
        evaluation_strategy="epoch",
        save_strategy="no",
        group_by_length=True,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'{args.selection}_a={aug}_c={c}_q={q}_m={args.m}_k={args.num_demonstration}_{args.seed}',
        max_grad_norm=0.3,
        remove_unused_columns=False,
        include_inputs_for_metrics=True,
    )

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    custom_loss = get_custom_loss_trainer()

    trainer = custom_loss(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=concatenate_datasets([dataset['min'], dataset['maj']]),
        eval_dataset={
            'min': dataset['min'], 
            'maj': dataset['maj']},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=prepare_compute_metrics(args.model_name.lower())
        )
    trainer.add_callback(CustomCallback(trainer))
    trainer.evaluate()


if __name__ == "__main__":
    args = get_args()
    main(args)