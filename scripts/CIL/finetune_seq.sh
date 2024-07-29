#!/bin/env/bash

set -e

eval "$(conda shell.bash hook)"
conda activate magmax

model=$1
dataset=$2
epochs=$3
n_splits=$4
seed=$5

# TRAIN
out_dir=outs/${model}/sequential_finetuning/class_incremental/ft/${dataset}
mkdir -p ${out_dir}

python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --sequential-finetuning \
    --seed ${seed} \
        |& tee ${out_dir}/splits:${n_splits}-ep:${epochs}-seed:${seed}.out


# MERGE
out_dir=outs/${model}/sequential_finetuning/class_incremental/merging/${dataset}
mkdir -p ${out_dir}

python merge_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --sequential-finetuning \
    --seed ${seed} \
        |& tee ${out_dir}/merge-${n_splits}-ep:${epochs}-seed:${seed}.out
