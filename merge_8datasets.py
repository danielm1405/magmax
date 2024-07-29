import torch
import numpy as np
import wandb
import os

from src.args import parse_arguments
from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_single_dataset, eval_task_aware


# Config
args = parse_arguments()

args.model = 'ViT-B-16'
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'


def search_evaluate_merging(datasets, task_vectors, args, n_coeffs=20):
    funcs_and_coeffs = [
        # (merge_rnd_mix, np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]),
        # (merge_max_abs, np.linspace(0.0, 1.0, num=n_coeffs+1)[1:]),
        # (sum, np.linspace(0.0, 2.0/args.n_splits, num=n_coeffs+1)[1:]),
        (merge_rnd_mix, [1.0]),
        (merge_max_abs, [0.5]),
        (sum, [1.0/args.n_splits]),
    ]

    for f, coeffs in funcs_and_coeffs:
        print(f"\nMerging with function: {f.__name__}")
        merged_tv = f(task_vectors)
        
        # Apply the resulting task vector
        results = {}
        for scaling_coef in coeffs:
            print(f"Scaling coeff: {scaling_coef}")
            image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            # Evaluate
            _r = {}
            print(datasets)
            for ds in datasets:
                _r[f"merging/{ds}/{f.__name__}"] = eval_single_dataset(image_encoder, ds, args)['top1'] * 100.0
                
            wandb.log({
                **_r,
                "helpers/merging/alpha": scaling_coef,
            })
            results[scaling_coef] = _r

        print(f"Results with function {f.__name__}:\n{results}")
        
    # TIES merging
    reset_type = 'topk'
    reset_thresh = 20
    resolve = 'mass'
    merge = 'dis-mean'
    tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
    
    print(f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}")
    
    merged_flat_tv = merge_methods(
        reset_type,
        tv_flat_checks,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )
    merged_tv = vector_to_state_dict(
        merged_flat_tv, task_vectors[0].vector, remove_keys=[]
    )
    merged_tv = TaskVector(vector=merged_tv)

    # Apply the resulting task vector
    results = {}
    # for scaling_coef in np.linspace(0.55, 1.5, num=n_coeffs+1):
    for scaling_coef in [0.55]:
        print(f"Scaling coeff: {scaling_coef}")
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        # Evaluate
        _r = {}
        for ds in datasets:
            _r[f"merging/{ds}/TIES"] = eval_single_dataset(image_encoder, ds, args)['top1'] * 100.0
            
        wandb.log({
            **_r,
            "helpers/merging/alpha": scaling_coef,
        })
        results[scaling_coef] = _r
            
    print(f"Results with function TIES:\n{results}")




if __name__ == '__main__':
    datasets = ['Cars', 'MNIST', 'EuroSAT', 'SVHN', 'RESISC45', 'SUN397', 'DTD', 'GTSRB']
    epochs = {
        'Cars': 35,
        'DTD': 75,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'ImageNet': 4
    }
    args.eval_datasets = [args.dataset]
    args.lr = 1e-5
    args.batch_size = 128
    args.n_splits = len(datasets)
    method = 'seq-ft' if args.sequential_finetuning else 'ind-ft'


    wandb.init(
        project="magmax",
        group=f"merging-8datasets",
        entity=args.wandb_entity_name,
        mode='online',
        name=f"merging-8datasets-{method}",
        config=args,
        tags=["merging", "8datasets", {method}],
    )
    
    if args.sequential_finetuning:
        base_path = f"checkpoints/{args.model}/8datasets/{'->'.join(datasets)}"
    else:
        base_path = f'checkpoints/{args.model}/8datasets/ind'

    task_vectors = []
    for ds in datasets:
        ft_path = os.path.join(base_path, f'{ds}/checkpoint-epochs:{epochs[ds]}-seed:{args.seed}.pt')
        tv = TaskVector(pretrained_checkpoint, ft_path)
        task_vectors.append(tv)

    print('='*100)
    print(f"\nEVAL 8 datasets")    
    print('='*100)
    search_evaluate_merging(datasets, task_vectors, args, n_coeffs=10)
