#!/bin/env/bash

set -e

eval "$(conda shell.bash hook)"
conda activate magmax

model=$1
dataset=$2
epochs=$3
n_splits=$4
ewc_lamb=$5
seed=$6


# TRAIN
out_dir=outs/${model}/ewc/class_incremental/ft/${dataset}
mkdir -p ${out_dir}

python ewc.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --ewc_lamb ${ewc_lamb} \
    --seed ${seed} \
        |& tee ${out_dir}/lamb:${ewc_lamb}-splits:${n_splits}-ep:${epochs}-seed:${seed}.out