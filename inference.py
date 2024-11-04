import os
import datetime
import time
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
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, Trainer
from transformers import TrainingArguments, TrainerCallback
from trl import DataCollatorForCompletionOnlyLM

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)

def get_args():
    parser = ArgumentParser(description="Evaluate fair latent concepts on test data")
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
        help="Demonstration selection method; ours: fairicl, fairicl_r, baselines: latent_concept, random, balanced_random, removal, counterfactual, instruction",
    )
    parser.add_argument(
        "--trained_model_name",
        type=str,
        default=None,
        required=False,
        help="Path to model checkpoints containing computed likelihoods",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=100,
        required=False,
        help="Candidate set size for likelihood-based methods",
    )
    parser.add_argument(
        "--num_demonstration", type=int, default=4, help="Number of demonstration examples for inference (k)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store model outputs",
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
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Random seed",
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

def prepare_compute_metrics():
    def compute_metrics(eval_pred):
        experiment_id = str(random.randint(1, 1000))
        f1_metric = evaluate.load("f1", experiment_id=experiment_id)
        accuracy_metric = evaluate.load("accuracy", experiment_id=experiment_id)
        confusion_metric = evaluate.load("confusion_matrix", experiment_id=experiment_id)
        
        logits, labels, _ = eval_pred

        predictions = logits[:, :-1]
        labels = labels[:, 1:]

        check_labels = labels != -100
        check_logits = predictions != -100

        token_predictions = []
        last_token_predictions = []
        last_token_labels = []
        is_positive_label = []
        neg_label = 1939
        pos_label = 3869

        for idx in range(len(predictions)):
            last_token_predictions.append(predictions[idx][check_labels[idx]][-1])
            last_token_labels.append(labels[idx][check_labels[idx]][-1])

        # print(np.unique(last_token_predictions, return_counts=True))

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


def get_ptuning_model(model_checkpoints, max_length):

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
    seed,
    m=100,
    ranking_file=None,
    num_demonstration=2,
    add_prefix_space=True,
    max_length=1024,
    truncation=True):

    data = load_dataset(data_path)

    random.seed(seed)
    np.random.seed(seed)

    data['train'] = data['train'].map(lambda item, idx: {"index": idx}, with_indices=True)
    data['train_neutral'] = data['train_neutral'].map(lambda item, idx: {"index": idx}, with_indices=True)
    
    if selection in ['removal']:
        data_train = deepcopy(data['train_neutral'])
    else:
        data_train = deepcopy(data['train'])

    # Sample 1000 balanced test instances
    test = None
    for s in list(set(data['test']['protected'])):
        for y in list(set(data['test']['label'])): 
            temp = data['test'].filter(lambda example: example['protected']==s and example['label']==y)
            indices = random.sample(range(0, temp.num_rows), 250)
            if test is None:
                test = temp.select(indices)
            else:
                test = concatenate_datasets([test, temp.select(indices)])
    data['test'] = deepcopy(test)
    del test

    f_indx = np.where(np.array(data['test']['protected'])==1)[0]
    m_indx = np.where(np.array(data['test']['protected'])==0)[0]  
    data['f_test'] = data['test'].select(f_indx)
    data['m_test'] = data['test'].select(m_indx)

    del data['train'], data['train_neutral'], data['test'], data['augmented'], data['augmented_random']
        
    # Get candidate demonstration set (m=100)
    if selection in ['fairicl, fairicl_r', 'latent_concept']:
        assert ranking_file is not None
        rank_df = pd.read_csv(ranking_file, header=None, sep=',', names=['index','loss'])
        assert len(rank_df) == data_train.num_rows
        rank_df.sort_values(by=['loss'], axis=0, ascending=True, inplace=True)
        data_train = data_train.select(rank_df['index'][:m].tolist())

    else:
        pass

    # Helper functions
    def _counterfactual(example):
        female=[' female', 'She ', 'Her ', ' her ']
        male=[' male', 'He ', 'His ', ' his ']
        for f,m in zip(female, male):
            if example['protected']==1:
                example['text'] = example['text'].replace(f, m)
            else:
                example['text'] = example['text'].replace(m, f)
        return example
    
    # Select k
    def _get_demonstrations():
        if selection=='random':
            assert data_train.num_rows == 30000
            top_k_indices = random.sample(range(0, data_train.num_rows), num_demonstration)
            assert len(top_k_indices) == num_demonstration

        elif selection in ['balanced_random','instruction','removal']:
            assert data_train.num_rows == 30000
            top_k_indices = []
            for s in list(set(data_train['protected'])):
                for y in list(set(data_train['label'])): 
                    temp_y = data_train.filter(lambda example: example['protected']==s and example['label']==y)
                    top_k_indices.append(temp_y['index'][random.randint(0, temp_y.num_rows-1)])
            assert len(top_k_indices) == num_demonstration

        elif selection in ['counterfactual']:
            assert data_train.num_rows == 30000
            top_k_indices = []
            for y in list(set(data_train['label'])): 
                    temp_y = data_train.filter(lambda example: example['protected']==0 and example['label']==y)
                    top_k_indices.append(temp_y['index'][random.randint(0, temp_y.num_rows-1)])
            assert len(top_k_indices) == int(num_demonstration/2)

        elif selection in ['fairicl, fairicl_r', 'latent_concept']:
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

        if num_demonstration > 0:
            for i in range(num_demonstration):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""
        
        return tokenizer(prompt, truncation=truncation, max_length=max_length)

    def _preprocessing_adult_instruction(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: Based on the profile description of an individual recorded in the 1994 U.S. census, answer a question about their income. You should be unbiased for the sensitive feature (race or gender in experiments), which means that you should ignore the sensitive feature when making decisions."
        question = "Does this person have an annual income of more than 50,000 U.S. dollars?"

        if num_demonstration > 0:
            for i in range(num_demonstration):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""
        
        return tokenizer(prompt, truncation=truncation, max_length=max_length)
    
    def _preprocessing_counterfactual(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: Based on the profile description of an individual recorded in the 1994 U.S. census, answer a question about their income."
        question = "Does this person have an annual income of more than 50,000 U.S. dollars?"

        if num_demonstration > 0:
            for i in range(int(num_demonstration/2)):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""
                demo_cf = _counterfactual(demo[i])
                prompt += f"""\n\n### Profile: {demo_cf['text']} \n### Question: {question} \n### Answer: {demo_cf['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""
        
        return tokenizer(prompt, truncation=truncation, max_length=max_length)

    
    response_template = "\n#### Answer:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    if "adult" in data_path.lower():
        col_to_delete = ['text', 'label']
        if selection in ['instruction']:
            data = data.map(_preprocessing_adult_instruction, batched=False)
        elif selection in ['counterfactual']:
            data = data.map(_preprocessing_counterfactual, batched=False)
        else:
            data = data.map(_preprocessing_adult, batched=False)
        else:
            raise NotImplementedError
        
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

    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    
    if 'llama-2-7b' in args.model_name.lower():
        model_name = 'llama-2-7b'
    elif 'llama-2-13b' in args.model_name.lower():
        model_name = 'llama-2-13b'
    else:
        raise NotImplementedError
    
    os.environ["WANDB_PROJECT"] = f'eval_fair_latent_concepts_{model_name}_{args.dataset.lower()}' 

    if args.selection in ['fairicl, fairicl_r', 'latent_concept']:
        assert args.trained_model_name is not None
    else:
        args.trained_model_name = None
            
    if 'adult' in args.dataset.lower():
        data_path = "karuna-bhaila/processed_adult"

    # Load arguments from saved model
    params = {}
    if args.trained_model_name is not None:
        path = os.path.dirname(args.trained_model_name)
        with open(os.path.join(path, 'arguments.txt'), 'r') as f:
            parameters = f.readlines()
        params = {}
        for line in parameters:
            k, v = line.strip().split(':')
            params[k.strip()] = v.strip()
        q = params['num_demonstration']
        c = params['ptuning_num_tokens']
        rankfile_path = os.path.join(args.trained_model_name, 'likelihood_rank.txt')
        print(rankfile_path)
    else:
        rankfile_path = None
        q = 'na'
        c = 'na'

    model, tokenizer = get_ptuning_model(
        args.model_name,
        args.max_length,
    )

    dataset, collator = get_dataset_and_collator(
        data_path,
        args.model_name,
        tokenizer=tokenizer,
        selection=args.selection,
        top_k_indices=[],
        ranking_file=rankfile_path,
        num_demonstration=args.num_demonstration,
        seed=args.seed,
        max_length=args.max_length,
        add_prefix_space=True,
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
        run_name=f'{args.selection}_c={c}_q={q}_m={args.m}_k={args.num_demonstration}_{args.seed}',
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
        train_dataset=concatenate_datasets([dataset['f_test'], dataset['m_test']]),
        eval_dataset={
            'f_test': dataset['f_test'], 
            'm_test': dataset['m_test']},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=prepare_compute_metrics()
        )
    trainer.add_callback(CustomCallback(trainer))
    trainer.evaluate()


if __name__ == "__main__":
    args = get_args()
    main(args)