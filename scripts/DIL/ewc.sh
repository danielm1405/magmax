#!/bin/env/bash

set -e

eval "$(conda shell.bash hook)"
conda activate magmax


model=$1
dataset=$2
epochs=$3
ewc_lamb=$4
seed=$5

# TRAIN
out_dir=outs/${model}/ewc/domain_incremental/ft/${dataset}
mkdir -p ${out_dir}

python ewc_domain_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --ewc_lamb ${ewc_lamb} \
    --seed ${seed} \
        |& tee ${out_dir}/ep:${epochs}-seed:${seed}.out
