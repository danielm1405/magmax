import os
import time
import torch
import wandb

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset, registry
from src.eval import eval_single_dataset, eval_given_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head

PRINT_EVERY = 100


def finetune(args, eval_0shot=False, only_eval_0shot=False):
    train_dataset_name = args.dataset
    dataset_class = registry[train_dataset_name].BASE_CLASS
    ckpdir = os.path.join(args.save, train_dataset_name)
    subset_config_id = dataset_class.get_md5(args.subset_config)

    # Check if checkpoints already exist
    if args.sequential_finetuning:
        ft_path = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{args.task_idx}.pt')
    else:
        ft_path = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{subset_config_id}.pt')
    if os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return

    assert train_dataset_name is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    elif args.sequential_finetuning and args.task_idx:
        prev_ckpt = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{args.task_idx-1}.pt')
        print(f'Loading image encoder from prev task {prev_ckpt=}')
        image_encoder = torch.load(prev_ckpt)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=True)

    preprocess_fn = image_encoder.train_preprocess
    
    # ZERO-SHOT EVAL ON EACH DOMAIN #
    if not args.skip_eval:
        wandb.log({'subset_config_ID': subset_config_id})
        
        if eval_0shot or only_eval_0shot:
            _full_r = eval_single_dataset(image_encoder, train_dataset_name, args)['top1']
            wandb.log({f'full_acc': _full_r * 100.0})

            for domain in dataset_class.DOMAINS:
                _subset_config = {
                    'domains': [domain],
                    'classes': dataset_class.CLASSES
                }
                _dataset = get_dataset(
                    train_dataset_name,
                    preprocess_fn,
                    location=args.data_location,
                    batch_size=args.batch_size,
                    subset_config=_subset_config,
                )
                _r = eval_given_dataset(image_encoder, _dataset, train_dataset_name, args)['top1']

                wandb.log({f'{domain}_acc': _r * 100.0})
        
    if only_eval_0shot:
        return
    ##################
    
    dataset = get_dataset(
        train_dataset_name,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        subset_config=args.subset_config,
    )
    
    if not args.skip_eval:
        wandb.log({
            'train_subset_samples': len(dataset.train_dataset),
            'test_subset_samples': len(dataset.test_dataset),
        })

    classification_head = get_classification_head(args, train_dataset_name, classnames=dataset.classnames)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    num_batches = len(dataset.train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        n_batches = len(data_loader)
        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % PRINT_EVERY == 0 or i + 1 == n_batches:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    if args.save is not None:
        image_encoder.save(ft_path)

    # FINETUNED EVAL ON EACH DOMAIN #
    if not args.skip_eval:
        _full_r = eval_single_dataset(image_encoder, train_dataset_name, args)['top1']
        wandb.log({f'full_acc': _full_r * 100.0})

        for domain in dataset_class.DOMAINS:
            _subset_config = {
                'domains': [domain],
                'classes': dataset_class.CLASSES
            }
            _dataset = get_dataset(
                train_dataset_name,
                preprocess_fn,
                location=args.data_location,
                batch_size=args.batch_size,
                subset_config=_subset_config,
            )
            _r = eval_given_dataset(image_encoder, _dataset, train_dataset_name, args)['top1']

            wandb.log({f'{domain}_acc': _r * 100.0})
    ##################


if __name__ == '__main__':

    args = parse_arguments()

    args.model = 'ViT-B-16'
    args.batch_size = 128
    sequential_ft_dir = 'sequential_finetuning/' if args.sequential_finetuning else ''
    args.save = f'checkpoints/{args.model}/{sequential_ft_dir}domain_incremental'
    dataset_class = registry[args.dataset]
    method = 'seq-ft' if args.sequential_finetuning else 'ind-ft'
    name=f"ft-{args.dataset}-DIL-{method}"

    for task_idx, domain_idx in enumerate(dataset_class.default_domain_order):
        args.subset_config = {
            'domains': [dataset_class.BASE_CLASS.DOMAINS[domain_idx]],
            'classes': dataset_class.BASE_CLASS.CLASSES,
        }
        
        args.task_idx = task_idx if args.sequential_finetuning else None

        if not args.skip_eval:
            wandb.init(
                project="magmax",
                group="ft-DIL",
                entity=args.wandb_entity_name,
                mode='online',
                name=f"{name}-" + ','.join(args.subset_config['domains']),
                config=args,
                reinit=True,
                tags=['ft', 'DIL', f"{args.dataset}", f"{method}"],
            )

        print('='*100)
        print(f'Finetuning {args.model} on {args.dataset}')
        print('='*100)

        finetune(args)
