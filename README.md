# FairICL: Fair In-Context Learning via Latent Concept Variables
This repository contains code for the paper Fair In-Context Learning via Latent Concept Variables.

 # Abstract
The emerging in-context learning (ICL) ability of large language models (LLMs) has prompted their use for predictive tasks in various domains with different types of data facilitated by serialization methods. However, with increasing applications in high-stakes domains, it has been shown that LLMs can inherit social bias and discrimination from their pre-training data. In this work, we investigate this inherent bias in LLMs during in-context learning with tabular data. We focus on an optimal demonstration selection approach that utilizes latent concept variables for resource-efficient task adaptation. We design data augmentation strategies that reduce correlation between predictive outcomes and sensitive variables helping to promote fairness during latent concept learning. We utilize the learned concept and select demonstrations from a training dataset to obtain fair predictions during inference while maintaining model utility. The latent concept variable is learned using a smaller internal LLM and the selected demonstrations can be used for inference with larger external LLMs. We empirically verify that the fair latent variable approach improves fairness results on tabular datasets compared to multiple heuristic demonstration selection methods.

## Datasets
We evaluate our method on the Adult Income Dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/2/adult). Please refer to the paper for details on constructions of the augmented dataset and teh serialization template. The following link points to the huggingface dataset containing the dataset splits in natural language format after serialization.
- [Adult Income](https://huggingface.co/datasets/karuna-bhaila/processed_adult)

## Installation and Usage
### Dependencies
```
# Install requirements
$ pip install -r requirements.txt
```
Our results are reported with Python==3.10.14. 

### Train
1. To learn (fair) latent concepts:
```
$ python train.py --dataset=adult --model_name=meta-llama/Llama-2-7b-hf --task=augmented_train --num_demonstration=2 \
      --ptuning_num_tokens=10 --num_epochs=5 --train_batch_size=32 --eval_batch_size=32 --augmented_size=1 --seed=42
```
- available task: train, augmented_train, random_train
- 
2. To compute likelihood using learned (fair) latent concepts:
```
$ python train_latent_concept.py --dataset=adult --model_name=[local path to checkpoints] --task=likelihood --num_demonstration=0 --eval_batch_size=32
```
`model_checkpoints` points to the checkpoints of a model obtained after training.

### Inference
1. To perform inference with top ranked demonstrations
```
$ python eval_fair_latent_concept.py --dataset=adult --model_name=meta-llama/Llama-2-13b-hf --trained_model_name=checkpoints/fair_latent_concepts_llama-2-7b_adult_2024-10-23-22-21-11/checkpoint-5860 --num_demonstration=4 --selection=likelihood --m=100 --eval_batch_size=8 --seed=1
```
`model_checkpoints` should point to the checkpoints of a model obtained after training with SPUL.
