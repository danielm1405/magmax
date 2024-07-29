#!/bin/env/bash

set -e

model=ViT-B-16
seed=5
lwf_lambda=0.3
ewc_lambda=1e6
epochs=10


for dataset in ImageNetR DomainNet ; do
    bash scripts/DIL/finetune.sh ${model} ${dataset} ${epochs} ${seed}
    bash scripts/DIL/finetune_seq.sh ${model} ${dataset} ${epochs} ${seed}
    bash scripts/DIL/ewc.sh ${model} ${dataset} ${epochs} ${ewc_lambda} ${seed}
    bash scripts/DIL/lwf.sh ${model} ${dataset} ${epochs} ${lwf_lambda} ${seed}
done
