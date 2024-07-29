import numpy as np
import torch
import wandb

from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_single_dataset, eval_task_aware, eval_task_agnostic
from src.args import parse_arguments


# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'


def evaluate_zero_shot(args, task_agnostic=True):
    _eval_f = eval_task_agnostic if task_agnostic else eval_task_aware
    _eval_name = "task-agnostic" if task_agnostic else "task-aware"
    
    print('#' * 100 + f"\nPerforming {_eval_name} evaluation of pretrained model.")

    print(f"\nEVAL: {args.dataset}-{args.n_splits} ({args.split_strategy} incremental) - zero-shot")

    # Create the task vectors
    image_encoder = torch.load(pretrained_checkpoint)

    # Evaluate
    res = _eval_f(image_encoder, args)
    print(f"{_eval_name} eval on {args.dataset} zero-shot. Accuracies:\n{res}\n" + '#' * 100 + '\n')


def evaluate_individial_fts(task_vectors, args, task_agnostic=True):
    _eval_f = eval_task_agnostic if task_agnostic else eval_task_aware
    _eval_name = "task-agnostic" if task_agnostic else "task-aware"
    
    print('#' * 100 + f"\nPerforming {_eval_name} evaluation of individual finetunings.")

    results = []
    for idx in range(args.n_splits):
        print(f"\nEVAL: {args.dataset}-{args.n_splits} ({args.split_strategy} incremental) - split idx: {idx}")

        # Create the task vectors
        image_encoder = task_vectors[idx].apply_to(pretrained_checkpoint, scaling_coef=1.0)

        # Evaluate
        res = _eval_f(image_encoder, args)
        results.append(res)
        print(f"{_eval_name} eval on {args.dataset} after task {idx}. Accuracies:\n{res}")

    print(f"{_eval_name} evaluation of individual finetunings: final results:\n{results}\n" + '#' * 100 + '\n')


def evaluate_merged_fts(task_vectors, args, merging_f, scaling_coef, task_agnostic=True, only_final=False):
    _eval_f = eval_task_agnostic if task_agnostic else eval_task_aware
    _eval_name = "task-agnostic" if task_agnostic else "task-aware"
    
    print('#' * 100 + f"\nPerforming {_eval_name} evaluation of merged finetunings.")

    results = []
    for idx in range(args.n_splits):
        if only_final and idx != args.n_splits - 1:
            continue
        
        print(f"\nEVAL: {args.dataset}-{args.n_splits} ({args.split_strategy} incremental) - split idx: {idx}")

        _tvs = task_vectors[:idx+1]
        merged_tv = merging_f(_tvs)
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        
        # Evaluate
        res = _eval_f(image_encoder, args)
        results.append(res)
        print(f"{_eval_name} eval on {args.dataset} after task {idx}. Accuracies:\n{res}")
    
    print(f"{_eval_name} evaluation of merged finetunings: final results:\n{results}\n" + '#' * 100 + '\n')


def search_evaluate_merging(task_vectors, dataset, n_splits, split_strategy, n_coeffs=20):
    print(f"\nEVAL: {dataset}-{n_splits} ({split_strategy} incremental)")
    
    funcs_and_coeffs = [
        # (merge_rnd_mix, np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]),
        # (merge_max_abs, np.linspace(0.0, 1.0, num=n_coeffs+1)[1:]),
        # (sum, np.linspace(0.0, 2.0/n_splits, num=n_coeffs+1)[1:]),
        (merge_rnd_mix, [1.0]),
        (merge_max_abs, [0.5]),
        (sum, [1.0/n_splits]),
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
            _r = eval_single_dataset(image_encoder, dataset, args)['top1']
            wandb.log({
                f"merging/{f.__name__}": _r * 100.0,
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
    # for scaling_coef in np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]:
    for scaling_coef in [0.55]:
        print(f"Scaling coeff: {scaling_coef}")
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        # Evaluate
        _r = eval_single_dataset(image_encoder, dataset, args)['top1']
        wandb.log({
            f"merging/TIES": _r * 100.0,
            "helpers/merging/alpha": scaling_coef,
        })
        results[scaling_coef] = _r
            
    print(f"Results with function TIES:\n{results}")


if __name__ == '__main__':
    
    sequential_ft_dir = 'sequential_finetuning/' if args.sequential_finetuning else ''
    args.save = f'checkpoints/{args.model}/ewc'
    
    # preload task vectors
    # task_vectors = [
    #     TaskVector(pretrained_checkpoint, f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}-lamb:0.2/finetuned_{_idx}.pt')
    #     for _idx in range(args.n_splits)
    # ]
    
    evaluate_zero_shot(args, task_agnostic=True)
    # evaluate_individial_fts(task_vectors, args, task_agnostic=False)
    # evaluate_individial_fts(task_vectors, args, task_agnostic=True)
    # evaluate_merged_fts(task_vectors, args, merge_max_abs, 0.5, task_agnostic=True)
    # evaluate_merged_fts(task_vectors, args, merge_max_abs, 0.5, task_agnostic=False)

    # search_evaluate_merging(task_vectors, args.dataset, args.n_splits, args.split_strategy)
