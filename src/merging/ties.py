import copy
from collections import OrderedDict
import torch
import logging


logger = logging.getLogger("root")


def merge_methods(
    reset_type,
    flat_task_checks,
    reset_thresh=None,
    resolve_method=None,
    merge_func="",
):
    all_checks = flat_task_checks.clone()

    if "nf" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = topk_mask_preserve_normfrac(
            all_checks, reset_thresh, return_mask=False
        )
    elif "topk" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = topk_values_mask(
            all_checks, K=reset_thresh, return_mask=False
        )
    elif "std" in reset_type and reset_thresh != "none":
        logger.info(f"Pruning: {reset_type} {reset_thresh}")
        updated_checks, *_ = greater_than_std_mask(
            all_checks, reset_thresh, return_mask=False
        )
    else:
        logger.info("Not removing NOISE")
        updated_checks = all_checks

    if resolve_method != "none":
        logger.info(f"RESOLVING SIGN: {resolve_method}")
        final_signs = resolve_sign(updated_checks, resolve_method)
        assert final_signs is not None
    else:
        logger.info("Not RESOLVING SIGN")
        final_signs = None

    if "dis" in merge_func:
        logger.info(f"Disjoint AGGREGATION: {merge_func}")
        merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    else:
        logger.info(f"Basic AGGREGATION: {merge_func}")
        merged_tv = aggregate(updated_checks, merge_func, final_signs)

    return merged_tv


### PRUNING ###

def topk_mask_preserve_normfrac(T, normfrac=0.9, return_mask=False):
    row_norms = torch.norm(T, p=2, dim=1, keepdim=True)

    # Calculate the proportion of each element's contribution to its row's norm
    proportion = T.abs() ** 2 / row_norms ** 2

    # Sort the proportions and their indices in descending order
    sorted_proportions, sorted_indices = torch.sort(proportion, dim=1, descending=True)

    # Calculate the cumulative sum of proportions
    cumsum_proportions = torch.cumsum(sorted_proportions, dim=1)

    # Find the indices where cumulative sum >= normfrac
    normfrac_mask = cumsum_proportions >= normfrac
    normfrac_indices = torch.argmax(normfrac_mask.float(), dim=1)

    # Create a range tensor to compare with normfrac_indices
    range_tensor = torch.arange(T.size(1)).unsqueeze(0).expand(T.size(0), -1)

    # Create a mask based on the normfrac_indices
    mask = range_tensor <= normfrac_indices.unsqueeze(1)

    # Initialize final_indices with a value that is out of bounds
    final_indices = torch.full_like(sorted_indices, T.size(1) - 1)

    # Use the mask to get the final indices
    final_indices[mask] = sorted_indices[mask]

    # Initialize the mask with zeros
    M = torch.zeros_like(T, dtype=torch.bool)

    # Use the final indices to update the final mask M
    M.scatter_(1, final_indices, True)

    if return_mask:
        return (T * M), M.float().mean(dim=1), M
    else:
        return (T * M), M.float().mean(dim=1)
    

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def bottomk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)

    # Create a mask tensor with True for the bottom k elements in each row
    mask = M.abs() <= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def greater_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() > factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


def less_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() < factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


### RESOLVING SIGN ###

def resolve_sign(Tensor, resolve_method):
    if resolve_method == "mass":
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
    elif resolve_method == "normfrac":
        sign_to_mult = normfrac_based_sign(Tensor)
    elif resolve_method == "normmass":
        sign_to_mult = normmass_based_sign(Tensor)
    else:
        raise ValueError(f"Sign resolve method {resolve_method} is not defined.")
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def normfrac_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return torch.sign(Tensor[norm_fracs.argmax(dim=0), torch.arange(Tensor.shape[1])])


def normmass_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return (Tensor.sign() * norm_fracs.abs()).sum(dim=0).sign()


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


### AGGREGATION ###

def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def aggregate(T, agg_type, final_signs, dim=0):
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    elif agg_type == "median":
        result = torch.median(T, dim=dim)[0]
    elif agg_type == "magnitude":
        max_indices = T.abs().argmax(dim=0)
        result = T[max_indices, torch.arange(T.shape[1])]
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)

    if final_signs is not None:
        # print(final_signs)
        result = result.abs() * final_signs

    return result


### UTILS ###

def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

