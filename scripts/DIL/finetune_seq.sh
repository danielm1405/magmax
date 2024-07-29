#!/bin/env/bash

set -e

eval "$(conda shell.bash hook)"
conda activate magmax

model=$1
dataset=$2
epochs=$3
seed=$4


# TRAIN
out_dir=outs/${model}/sequential_finetuning/domain_incremental/ft/${dataset}
mkdir -p ${out_dir}

python finetune_domain_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --sequential-finetuning \
        |& tee ${out_dir}/ep:${epochs}-seed:${seed}.out


# MERGE
out_dir=outs/${model}/sequential_finetuning/domain_incremental/merging/${dataset}
mkdir -p ${out_dir}

python merge_domain_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --sequential-finetuning \
        |& tee ${out_dir}/ep:${epochs}-seed:${seed}.out
