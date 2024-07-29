#!/bin/env/bash

set -e

eval "$(conda shell.bash hook)"
conda activate magmax


model=$1
dataset=$2
epochs=$3
lwf_lamb=$4
seed=$5

# TRAIN
out_dir=outs/${model}/lwf/domain_incremental/ft/${dataset}
mkdir -p ${out_dir}

python lwf_domain_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --lwf_lamb ${lwf_lamb} \
    --seed ${seed} \
        |& tee ${out_dir}/ep:${epochs}-seed:${seed}.out
