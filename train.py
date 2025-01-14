import os
import pickle
import datetime
import time
import sys
import random
import numpy as np
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
from transformers.utils import is_sagemaker_mp_enabled
from trl import DataCollatorForCompletionOnlyLM

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)

def get_args():
    parser = ArgumentParser(description="Train latent concepts using augmented data to promote fairness")
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
        help="Name of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--task",
        type=str,
        default='train',
        required=True,
        help="Whether to train latet concept tokens or compute likelihood"
    )
    parser.add_argument(
        "--augmented_size",
        type=float,
        default=0.0,
        required=False,
        help="size of augmented dataset relative to original training data"
    ),
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Random seed"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store the fine-tuned model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Eval batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument(
        "--num_tokens", type=int, default=10, help="Number of learnable tokens (c)"
    )
    parser.add_argument(
        "--num_demonstration", type=int, default=2, help="Number of demonstration examples (q)"
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

        f1 = f1_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='macro')["f1"]
        accuracy = accuracy_metric.compute(predictions=last_token_predictions, references=last_token_labels)["accuracy"]
        cm = confusion_metric.compute(predictions=last_token_predictions, references=last_token_labels, labels=[neg_label, pos_label])['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        return {'accuracy': accuracy, 'f1-score': f1, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

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


def get_model_tokenizer(model_checkpoints, max_length, num_tokens):

    base_model = AutoModelForCausalLM.from_pretrained(
        model_checkpoints, 
        device_map="auto", 
        offload_folder="offload", 
        trust_remote_code=True, 
        )

    if 'fair_latent_concepts' in model_checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoints, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    else:
        # Add new tokens corresponding to latent concepts
        new_tokens = [f'token{i}' for i in range(num_tokens)]
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoints, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            additional_special_tokens=new_tokens
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        base_model.config.original_vocab_size = deepcopy(base_model.config.vocab_size)

        # Resize embedding matrix
        base_model.resize_token_embeddings(len(tokenizer))
        print("Original vocab size: ", base_model.config.original_vocab_size)
        print(f'New vocab size: {base_model.config.vocab_size}')

    model = base_model

    for n, p in model.named_parameters():
        if p.requires_grad:
            if "embed" not in n:
                print(f"Turning {n} to untrainable")
                p.requires_grad = False
    
    summary(model)
    model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


def get_dataset_and_collator(
    data_path,
    model_checkpoints,
    tokenizer,
    mode='train',
    augmented_size=1,
    num_demonstration=2,
    seed=42,
    max_length=1024,
    truncation=True,):

    data = load_dataset(data_path)

    if mode in ['random_train']:
        assert 'adult' in data_path.lower()
        data['augmented'] = deepcopy(data['augmented_random'])
        
    if augmented_size is not None and augmented_size < 1:
        nrows = int(augmented_size*data['train'].num_rows)
        data['augmented'] = data['augmented'].select(random.sample(range(0, data['train'].num_rows), nrows))
    
    if 'train_neutral' in data.keys():
        del data['train_neutral']
    if 'augmented_random' in data.keys():
        del data['augmented_random']

    data['train'] = data['train'].map(lambda item, idx: {"index": idx}, with_indices=True)
    data['augmented'] = data['augmented'].map(lambda item, idx: {"index": idx}, with_indices=True)
    data['test'] = data['test'].map(lambda item, idx: {"index": idx}, with_indices=True)

    data_train = deepcopy(data['train'])
    
    def _get_demonstrations():
        return data_train.select(random.sample(range(0, data_train.num_rows), num_demonstration))
    
    def _preprocessing_adult(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: Based on the profile description of an individual recorded in the 1994 U.S. census, answer a question about their income."
        question = "Does this person have an annual income of more than 50,000 U.S. dollars?"

        if num_demonstration > 0:
            for i in range(num_demonstration):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""

        return tokenizer(prompt, truncation=truncation, max_length=max_length)

    def _preprocessing_compas(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: The profile of a criminal defendant includes demographic information and crime statistics. It also includes a potential risk computed based on the defendant's response to a COMPAS screening survey. Using the given information, determine whether the defendant recidivates in two years."
        question = "Does the defendant recidivate in two years?"

        if num_demonstration > 0:
            for i in range(num_demonstration):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: {question} \n### Answer: {demo[i]['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: {question} \n#### Answer: {example['label']}"""

        return tokenizer(prompt, truncation=truncation, max_length=max_length)
    
    def _preprocessing_lawschool(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: Based on the information provided about a law school student, determine whether the student passes the bar exam."
        question = "Does the student pass the bar exam on the first try?"

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

    def _append_latent_concepts(example):
        latent_concept_tokens = tokenizer.additional_special_tokens_ids
        latent_concept_mask = [1 for _ in latent_concept_tokens]
        padding_len = max_length - (len(example['input_ids']) + len(latent_concept_tokens))

        return {
            'input_ids': example['input_ids']+latent_concept_tokens,
            'attention_mask': example['attention_mask']+latent_concept_mask,
            'latent_concept_mask': [0 for _ in example['input_ids']]+[1 for _ in latent_concept_tokens]+[0 for _ in range(padding_len)]
        }

    response_template = "\n#### Answer:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    col_to_delete = ['text', 'label']

    if mode in ['train', 'augmented_train', 'random_train']:
        if mode in ['train']:
            del data['augmented']
        if "adult" in data_path.lower():
            data = data.map(_preprocessing_adult, batched=False)
        elif "compas" in data_path.lower():
            data = data.map(_preprocessing_compas, batched=False)
        elif "law" in data_path.lower():
            data = data.map(_preprocessing_lawschool, batched=False)
        data = data.map(_prepend_latent_concepts, batched=False)

    elif mode=='likelihood':
        num_demonstration = 0
        del data['augmented'], data['test']
        if "adult" in data_path.lower():
            data = data.map(_preprocessing_adult, batched=False)
        elif "compas" in data_path.lower():
            data = data.map(_preprocessing_compas, batched=False)
        elif "law" in data_path.lower():
            data = data.map(_preprocessing_lawschool, batched=False)
        data = data.map(_append_latent_concepts, batched=False)

    data = data.remove_columns(col_to_delete)
    data.set_format("torch")

    print(data)

    return data, data_collator


def get_latent_concept_loss_trainer():
    class LatentConceptTrainer(Trainer):
        def __init__(self, num_virtual_tokens, mode='train', **kwargs):
            super().__init__(**kwargs)
            self.num_virtual_tokens = num_virtual_tokens
            self.mode = mode

        def training_step(self, model, inputs):
            model.train()
            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            del inputs

            kwargs = {}

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss, **kwargs)

            # manually convert grads of original vocab to 0
            model.get_input_embeddings().weight.grad[:self.model.config.original_vocab_size] = 0

            return loss.detach() / self.args.gradient_accumulation_steps

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            if "index" not in inputs.keys():
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
                return (loss, outputs) if return_outputs else loss
            else:
                sample_indices = inputs.pop("index")
                sensitive_attr = inputs.pop("protected")
                if 'latent_concept_mask' in inputs.keys():
                    latent_concept_mask = inputs.pop("latent_concept_mask")

                # forward pass input+learnable_prompt
                outputs = model(**inputs)
                logits = outputs.get("logits")

                # Shift output by one to the right so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = ce_loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
                loss = torch.sum(loss) / len(sample_indices)

                return (loss, outputs) if return_outputs else loss

        def compute_likelihood(self, model, dataset):
            model.eval()
            dataloader = self.get_test_dataloader(dataset)
            likelihood = []
            indices = []
            
            for i, inputs in enumerate(tqdm(dataloader)):
                sample_indices = inputs.pop("index")
                if 's_index' in inputs.keys():
                    _ = inputs.pop("index") 
                labels = inputs.pop('labels')
                sensitive_attr = inputs.pop("protected")
                latent_concept_mask = inputs.pop("latent_concept_mask")
                latent_concept_mask = latent_concept_mask[:, :inputs['attention_mask'].shape[1]]

                # forward pass concept tokens+input
                outputs = model(**inputs)
                logits = outputs.get("logits")

                # Shift output by one to the right so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_latent_concept_mask = latent_concept_mask[..., 1:].contiguous()

                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = ce_loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

                loss = loss.view(shift_logits.size(0), shift_logits.size(1)) * shift_latent_concept_mask
                loss = torch.sum(loss, axis=1) / torch.sum(shift_latent_concept_mask, axis=1)

                indices.extend(sample_indices.detach().cpu().numpy().tolist()) 
                likelihood.extend(loss.detach().cpu().numpy().tolist()) 

            return indices, likelihood

    return LatentConceptTrainer


def main(args):
    # Random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    
    if 'llama-2-7b' in args.model_name.lower():
        model_name = 'llama-2-7b'
    elif 'llama-3-8b' in args.model_name.lower():
        model_name = 'llama-3-8b'
    else:
        raise NotImplementedError

    os.environ["WANDB_PROJECT"] = f'train_fair_latent_concepts_{model_name}_{args.dataset.lower()}' 
    
    if 'adult' in args.dataset.lower():
        data_path = "karuna-bhaila/processed_adult"
    elif 'compas' in args.dataset.lower():
        data_path = "karuna-bhaila/processed_compas"
    elif 'law' in args.dataset.lower():
        data_path = "karuna-bhaila/processed_lawschool"

    # Specify paths
    if args.output_path is None and 'train' in args.task:
        args.output_path = f'checkpoints/fair_latent_concepts_{model_name}_{args.dataset.lower()}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        # write run arguments to file
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                f.write(f'{k}: {v}\n')

    model, tokenizer = get_model_tokenizer(
        args.model_name,
        args.max_length,
        args.num_tokens,
    )
                
    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        group_by_length=True,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'{args.task}_lr={args.lr}_aug={args.augmented_size}_numtokens={args.num_tokens}_q={args.num_demonstration}',
        max_grad_norm=0.3,
        remove_unused_columns=False,
        include_inputs_for_metrics=True,
    )

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    custom_loss = get_latent_concept_loss_trainer()

    dataset, collator = get_dataset_and_collator(
        data_path,
        args.model_name,
        tokenizer=tokenizer,
        mode=args.task,
        augmented_size=args.augmented_size,
        num_demonstration=args.num_demonstration,
        seed=args.seed,
        max_length=args.max_length,
        truncation=True,
    )

    # Train model
    if 'train' in args.task:
        if args.task in ['augmented_train', 'random_train']:
            dataset['train'] = concatenate_datasets([dataset['train'], dataset['augmented']])

        trainer = custom_loss(
            model=model,
            num_virtual_tokens=args.num_tokens,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset={
                "org_train": dataset['train'],
                "test": dataset['test']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=prepare_compute_metrics(args.model_name.lower())
            )
        trainer.add_callback(CustomCallback(trainer))
        trainer.train()

    # Compute likelihood
    elif args.task=='likelihood':
        training_args.output_dir = 'inference/temp'
        trainer = custom_loss(
            model=model,
            num_virtual_tokens=args.num_tokens,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=prepare_compute_metrics(args.model_name.lower())
            )
        trainer.add_callback(CustomCallback(trainer))
        indices, likelihood = trainer.compute_likelihood(trainer.model, dataset['train'])

        # save to file
        filepath = os.path.join(args.model_name, 'likelihood_rank.txt')
        with open(filepath, 'w') as f:
            for idx, val in zip(indices, likelihood):
                f.write(f'{idx},{val}\n')


if __name__ == "__main__":
    args = get_args()
    main(args)
    