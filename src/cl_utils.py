import torch
from torch.utils.data.dataset import Subset

from src.datasets.common import (
    get_balanced_data_incremental_subset_indices,
    get_class_incremental_classes_and_subset_indices,
)
from src.heads import get_classification_head, build_subset_classification_head


def get_dataset_and_classifier_for_split(dataset, split_idx, text_encoder, args, remap_labels=True, return_classifier=True):
    if args.split_strategy == 'data':
        train_subset_indices, test_subset_indices = \
            get_balanced_data_incremental_subset_indices(
                dataset.train_dataset, args.n_splits, split_idx
            )
        dataset.train_dataset = torch.utils.data.Subset(dataset.train_dataset, train_subset_indices)
        # it does not make sense to split test in data-incremental
        # dataset.test_dataset = torch.utils.data.Subset(dataset.test_dataset, test_subset_indices)
        if return_classifier:
            classification_head = get_classification_head(args, args.dataset)
    elif args.split_strategy == 'class':
        classes, train_subset_indices, test_subset_indices = \
            get_class_incremental_classes_and_subset_indices(
                dataset, args.n_splits, split_idx
            )

        dataset.train_dataset = Subset(dataset.train_dataset, train_subset_indices)
        dataset.test_dataset = Subset(dataset.test_dataset, test_subset_indices)

        if remap_labels:
            class_map = {c: idx for idx, c in enumerate(sorted(classes))}        
            dataset.train_dataset.dataset.target_transform = lambda t : class_map[t]
            dataset.test_dataset.dataset.target_transform = lambda t : class_map[t]

        if return_classifier:
            classification_head = build_subset_classification_head(
                text_encoder.model, args.dataset, classes, args.data_location, args.device
            )
    else:
        raise NotImplementedError()
    
    # dataloaders
    dataset.train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset, batch_size=dataset.train_loader.batch_size,
        shuffle=True, num_workers=dataset.train_loader.num_workers
    )        
    dataset.test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset, batch_size=dataset.test_loader.batch_size,
        shuffle=False, num_workers=dataset.test_loader.num_workers
    )

    return (dataset, classification_head) if return_classifier else dataset
        