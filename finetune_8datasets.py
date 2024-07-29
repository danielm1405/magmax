import os
import time
import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head



def finetune(args):
    train_dataset = args.dataset
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if checkpoints already exist
    ft_path = os.path.join(ckpdir, f'checkpoint-epochs:{args.epochs}-seed:{args.seed}.pt')
    if os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        print(f'Loading previous image encoder {args.load}.')
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=True)

    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )

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

            if step % print_every == 0 or i + 1 == n_batches:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)

    if args.save is not None:
        image_encoder.save(ft_path)

    return ft_path


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
    args = parse_arguments()

    args.model = 'ViT-B-16'
    args.lr = 1e-5
    args.batch_size = 128
    args.load = None
    
    if args.sequential_finetuning:
        args.save = f"checkpoints/{args.model}/8datasets/{'->'.join(datasets)}"
    else:
        args.save = f'checkpoints/{args.model}/8datasets/ind'

    for dataset in datasets:
        args.dataset = dataset
        args.epochs = epochs[args.dataset]
        args.eval_datasets = [args.dataset]

        print('='*100)
        print(f'Finetuning {args.model} on {args.dataset}')
        print('='*100)

        last_ckpt = finetune(args)
        if args.sequential_finetuning:
            args.load = last_ckpt
