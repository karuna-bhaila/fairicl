import os
import datetime
import time
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
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainerState, TrainerControl, AutoModelForCausalLM, Trainer
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
        help="Whether to train latent concept tokens or compute likelihood"
    )
    parser.add_argument(
        "--augmented_size",
        type=float,
        default=1.0,
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
        "--ptuning_num_tokens", type=int, default=10, help="Number of learnable tokens (c)"
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

def prepare_compute_metrics(tokenizer, mode=None):
    def compute_metrics(eval_pred):
        nonlocal tokenizer
        nonlocal mode

        experiment_id = str(random.randint(1, 1000))
        f1_metric = evaluate.load("f1", experiment_id=experiment_id)
        accuracy_metric = evaluate.load("accuracy", experiment_id=experiment_id)
        confusion_metric = evaluate.load("confusion_matrix", experiment_id=experiment_id)
        precision_metric = evaluate.load('precision', experiment_id=experiment_id)
        recall_metric = evaluate.load('recall', experiment_id=experiment_id)

        logits, labels, inputs = eval_pred

        predictions = logits[:, :-1]
        labels = labels[:, 1:]

        check_labels = labels != -100
        check_logits = predictions != -100

        token_predictions = []
        last_token_predictions = []
        last_token_labels = []

        neg_label = 1939   # No
        pos_label = 3869   # Yes

        for idx in range(len(predictions)):
            last_token_predictions.append(predictions[idx][check_labels[idx]][-1])
            last_token_labels.append(labels[idx][check_labels[idx]][-1])

        # print(np.unique(last_token_predictions, return_counts=True))

        f1 = f1_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='macro')["f1"]
        accuracy = accuracy_metric.compute(predictions=last_token_predictions, references=last_token_labels)["accuracy"]
        cm = confusion_metric.compute(predictions=last_token_predictions, references=last_token_labels, labels=[neg_label, pos_label])['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        if mode=='fairicl_lc':
            sp = (tp+fp) / (tn+fp+fn+tp)
            tpr = tp / (tp+fn)
            fpr = fp / (tn+fp)

            print(sp, tpr)
            
            return {'acc': accuracy, 'f1': f1, 'sp': sp, 'tpr': tpr, 'fpr': fpr, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        
        else:
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


def get_ptuning_model(model_checkpoints, max_length, num_tokens):

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
    add_prefix_space=True,
    max_length=1024,
    truncation=True,):

    random.seed(seed) 
    np.random.seed(seed)

    data = load_dataset(data_path)

    if mode in ['random_train']:
        data['augmented'] = deepcopy(data['augmented_random'])
    del data['train_neutral'], data['augmented_random']
        
    if augmented_size < 1:
        nrows = int(augmented_size*data['train'].num_rows)
        data['augmented'] = data['augmented'].select(random.sample(range(0, data['train'].num_rows), nrows))

    data['train'] = data['train'].map(lambda item, idx: {"index": idx}, with_indices=True)
    data['augmented'] = data['augmented'].map(lambda item, idx: {"index": idx}, with_indices=True)
    data['test'] = data['test'].map(lambda item, idx: {"index": idx}, with_indices=True)

    data_train = deepcopy(data['train'])
    
    def _get_demonstrations():
        return data_train.select(random.sample(range(0, data_train.num_rows), num_demonstration))
    
    def _preprocessing_adult(example):
        demo = _get_demonstrations()
        prompt = "### Instruction: Based on the profile description of an individual recorded in the 1994 U.S. census, answer a question about their income."

        if num_demonstration > 0:
            for i in range(num_demonstration):
                prompt += f"""\n\n### Profile: {demo[i]['text']} \n### Question: Does this person have an annual income of more than 50,000 U.S. dollars? \n### Answer: {demo[i]['label']}"""

        prompt += f"""\n\n### Profile: {example['text']}  \n### Question: Does this person have an annual income of more than 50,000 U.S. dollars? \n#### Answer: {example['label']}"""

        return tokenizer(prompt, truncation=truncation, max_length=max_length)

    def _preprocessing_likelihood(data):
        sampled_indices = []
        _indices = data['index'] + data['index']
        for indx in tqdm(range(data.num_rows)):
            _samples = []
            for _ in range(num_demonstration):
                while True:
                    rand_indx = random.sample(range(0, len(_indices)), 1)[0]
                    if _indices[rand_indx] not in _samples:
                        _samples.append(_indices[rand_indx])
                        _indices.pop(rand_indx)
                        break
            
            sampled_indices.append([_samples[0], _samples[1]])

        data = data.map(lambda item, idx: {"s_index": sampled_indices[idx]}, with_indices=True)
        return data    

    def _tokenize(example):
        indices = example['s_index']
        prompt = "### Instruction: Based on the profile description of an individual recorded in the 1994 U.S. census, answer a question about their income."

        if num_demonstration > 1:
            for i in range(num_demonstration-1):
                prompt += f"""\n\n### Profile: {data_train[indices[i]]['text']} \n### Question: Does this person have an annual income of more than 50,000 U.S. dollars? \n### Answer: {data_train[indices[i]]['label']}"""

        prompt += f"""\n\n### Profile: {data_train[indices[-1]]['text']}  \n### Question: Does this person have an annual income of more than 50,000 U.S. dollars? \n#### Answer: {data_train[indices[-1]]['label']}"""
        
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

    if "adult" in data_path.lower():
        col_to_delete = ['text', 'label']

        if mode in ['train', 'resume_train', 'augmented_train', 'random_train']:
            data = data.map(_preprocessing_adult, batched=False)
            data = data.map(_prepend_latent_concepts, batched=False)

        elif mode=='likelihood':
            num_demonstration = 0
            del data['augmented'], data['test']
            data = data.map(_preprocessing_adult, batched=False)
            data = data.map(_append_latent_concepts, batched=False)
            
        elif mode=='fairicl_lc':
            del data['train'], data['augmented']
            data = data.map(_preprocessing_adult, batched=False)
            data = data.map(_prepend_latent_concepts, batched=False)
            # Sample 1000 balanced test instances
            random.seed(seed)
            np.random.seed(seed)
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
            del data['test']

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

                # print(likelihood)

            return indices, likelihood

    return LatentConceptTrainer


def main(args):
    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    
    if 'llama-2-7b' in args.model_name.lower():
        model_name = 'llama-2-7b'
    elif 'llama-2-13b' in args.model_name.lower():
        model_name = 'llama-2-13b'
    else:
        raise NotImplementedError

    os.environ["WANDB_PROJECT"] = f'train_fair_latent_concepts_{model_name}_{args.dataset.lower()}' 
    
    if 'adult' in args.dataset.lower():
        data_path = "karuna-bhaila/processed_adult"

    # Specify paths
    if args.output_path is None:
        args.output_path = f'checkpoints/fair_latent_concepts_{model_name}_{args.dataset.lower()}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        # write run arguments to file
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                f.write(f'{k}: {v}\n')

    model, tokenizer = get_ptuning_model(
        args.model_name,
        args.max_length,
        args.ptuning_num_tokens,
    )
                
    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="no",
        save_strategy="epoch",
        group_by_length=True,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'lr={args.lr}_aug={args.augmented_size}_numtokens={args.ptuning_num_tokens}_q={args.num_demonstration}',
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
        add_prefix_space=True,
        truncation=True,
    )

    if args.task=='train':
        trainer = custom_loss(
            model=model,
            num_virtual_tokens=args.ptuning_num_tokens,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['train'],
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=prepare_compute_metrics(tokenizer)
            )
        trainer.add_callback(CustomCallback(trainer))

        trainer.train()

    elif args.task in ['augmented_train','random_train']:
        dataset['joint_train'] = concatenate_datasets([dataset['train'], dataset['augmented']])
        trainer = custom_loss(
            model=model,
            num_virtual_tokens=args.ptuning_num_tokens,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['joint_train'],
            eval_dataset={
                "org_train": dataset['train']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=prepare_compute_metrics(tokenizer)
            )
        trainer.add_callback(CustomCallback(trainer))

        trainer.train()
    
    elif args.task=='likelihood':
        training_args.output_dir = f'inference/temp'
        trainer = custom_loss(
            model=model,
            num_virtual_tokens=args.ptuning_num_tokens,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=prepare_compute_metrics(tokenizer)
            )
        trainer.add_callback(CustomCallback(trainer))
        indices, likelihood = trainer.compute_likelihood(trainer.model, dataset['train'])

        # save to file
        filepath = os.path.join(args.model_name, 'likelihood_rank.txt')
        with open(filepath, 'w') as f:
            for idx, val in zip(indices, likelihood):
                f.write(f'{idx},{val}\n')

    elif args.task=='fairicl_lc':
        # Load arguments from saved model
        path = os.path.dirname(args.model_name)
        with open(os.path.join(path, 'arguments.txt'), 'r') as f:
            parameters = f.readlines()
        params = {}
        for line in parameters:
            k, v = line.strip().split(':')
            params[k.strip()] = v.strip()
        q = params['num_demonstration']
        c = params['ptuning_num_tokens']
        aug = params['augmented_size']
        os.environ["WANDB_PROJECT"] = f'eval_fair_latent_concepts_{model_name}_{args.dataset.lower()}' 
        training_args.run_name = f'fairicl_lc_c={c}_q={q}_k={args.num_demonstration}_{args.seed}'
        
        trainer = custom_loss(
            model=model,
            num_virtual_tokens=args.ptuning_num_tokens,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=concatenate_datasets([dataset['f_test'], dataset['m_test']]),
            eval_dataset={
                'f_test': dataset['f_test'], 
                'm_test': dataset['m_test']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=prepare_compute_metrics(tokenizer, args.task)
            )
        trainer.add_callback(CustomCallback(trainer))
        trainer.evaluate()


if __name__ == "__main__":
    args = get_args()
    main(args)
