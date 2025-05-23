# FairICL: Fair In-Context Learning via Latent Concept Variables
This repository contains code for the paper Fair In-Context Learning via Latent Concept Variables.

 # Abstract
The emerging in-context learning (ICL) ability of large language models (LLMs) has prompted their use for predictive tasks in various domains with different datatypes, including tabular data, facilitated by serialization methods. However, with increasing applications in high-stakes domains, it has been shown that LLMs can inherit social bias and discrimination from their pre-training data. In this work, we investigate this inherent bias in LLMs during in-context learning with tabular data. We focus on an optimal demonstration selection approach that utilizes latent concept variables for resource-efficient task adaptation. We design data augmentation strategies that reduce correlation between predictive outcomes and sensitive variables helping to promote fairness during latent concept learning. We utilize the learned concept and select demonstrations from a training dataset to obtain fair predictions during inference while maintaining model utility. The latent concept variable is learned using a smaller internal LLM and the selected demonstrations can be used for inference with larger external LLMs. We empirically verify that the fair latent variable approach improves fairness results on tabular datasets compared to multiple heuristic demonstration selection methods.

## Datasets
We evaluate our method on three datasets listed below. Please refer to the paper for details on constructions of the augmented dataset and the serialization template. [This link](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy9jNjVmNmM5ZTM0NTY2MWUzL0VqU3dyNVRKckFKRWlsSF9CTEtPWHFJQkhuOHFnQ3R0bGpoN29PWk9VVG5QdFE%5FZT1CM1Q0bGg&id=C65F6C9E345661E3%21se4b54e9727b34dc39b502884c9887601&cid=C65F6C9E345661E3&sb=name&sd=1) points to the huggingface dataset containing the dataset splits in natural language format after serialization.

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
$ python train.py --dataset=adult --model_name=[model checkpoints or name in hf] --task=augmented_train --num_demonstration=2 --ptuning_num_tokens=10 --num_epochs=5 --train_batch_size=32 --eval_batch_size=32 --augmented_size=1 --seed=42
```
- available tasks: train (learn latent concepts), augmented_train (learn fair latent concepts)

2. To compute likelihood using learned (fair) latent concepts:
```
$ python train.py --dataset=adult --model_name=[local path to checkpoints] --task=likelihood --num_demonstration=0 --eval_batch_size=32
```
`model_name` should point to the checkpoints of a model obtained after latent concept learning.

### Inference
1. To perform inference with top-ranked demonstrations
```
$ python inference.py --dataset=adult --model_name=[model checkpoints or name in hf] --trained_model_name=[local path to checkpoints containing likelihood file] --num_demonstration=4 --selection=fairicl --m=100 --eval_batch_size=8 --seed=1
```
`model_name` refers to the external LLM for inference and `trained_model_name` should point to the checkpoints of the internal llm obtained after likelihood computation.

- available selection methods: fairicl, latent_concept, random, balanced, removal, counterfactual, instruction
