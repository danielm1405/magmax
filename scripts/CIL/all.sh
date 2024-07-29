#!/bin/env/bash

set -e

model=ViT-B-16
seed=5
lwf_lambda=0.3
ewc_lambda=1e6

epochs=10
dataset=CIFAR100
for n_splits in 5 10 20 50 ; do
    bash scripts/CIL/finetune.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed}
    bash scripts/CIL/finetune_seq.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed}
    bash scripts/CIL/ewc.sh ${model} ${dataset} ${epochs} ${n_splits} ${ewc_lambda} ${seed}
    bash scripts/CIL/lwf.sh ${model} ${dataset} ${epochs} ${n_splits} ${lwf_lambda} ${seed}
done

epochs=10
dataset=ImageNetR
for n_splits in 5 10 20 50 ; do
    bash scripts/CIL/finetune.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed}
    bash scripts/CIL/finetune_seq.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed}
    bash scripts/CIL/ewc.sh ${model} ${dataset} ${epochs} ${n_splits} ${ewc_lambda} ${seed}
    bash scripts/CIL/lwf.sh ${model} ${dataset} ${epochs} ${n_splits} ${lwf_lambda} ${seed}
done

epochs=30
dataset=CUB200
for n_splits in 5 10 20 ; do
    bash scripts/CIL/finetune.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed}
    bash scripts/CIL/finetune_seq.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed}
    bash scripts/CIL/ewc.sh ${model} ${dataset} ${epochs} ${n_splits} ${ewc_lambda} ${seed}
    bash scripts/CIL/lwf.sh ${model} ${dataset} ${epochs} ${n_splits} ${lwf_lambda} ${seed}
done

epochs=30
dataset=Cars
for n_splits in 5 10 20 ; do
    bash scripts/CIL/finetune.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed}
    bash scripts/CIL/finetune_seq.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed}
    bash scripts/CIL/ewc.sh ${model} ${dataset} ${epochs} ${n_splits} ${ewc_lambda} ${seed}
    bash scripts/CIL/lwf.sh ${model} ${dataset} ${epochs} ${n_splits} ${lwf_lambda} ${seed}
done
