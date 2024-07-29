#!/bin/env/bash

set -e

model=ViT-B-16
seed=5

bash scripts/8ds/finetune_8ds.sh ${model} ${seed}
bash scripts/8ds/finetune_8ds_seq.sh ${model} ${seed}
