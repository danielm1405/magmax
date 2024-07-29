import os
import time
import torch
from copy import deepcopy

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier, concat_classification_heads
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.eval import evaluate


def cross_entropy(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    """Calculates cross-entropy with temperature scaling"""
    out = torch.nn.functional.softmax(outputs, dim=1)
    tar = torch.nn.functional.softmax(targets, dim=1)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce
        

def finetune(args):
    train_dataset = args.dataset
    
    # finetune for each split separately
    for split_idx in range(args.n_splits):
        print(f"\n##### SPLIT {split_idx} #####")
        ckpdir = os.path.join(args.save, f"{train_dataset}-{args.n_splits}", f"ft-epochs-{args.epochs}-seed:{args.seed}-lamb:{args.lwf_lamb}")
        ft_path = os.path.join(ckpdir, f'finetuned_{split_idx}.pt')

        if split_idx == 0:
            print('Building image encoder.')
            image_encoder = ImageEncoder(args, keep_lang=True)

        preprocess_fn = image_encoder.train_preprocess
        print_every = 10

        dataset = get_dataset(
            train_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        dataset, classification_head = get_dataset_and_classifier_for_split(
            dataset, split_idx, image_encoder, args
        )
        
        model = ImageClassifier(image_encoder, classification_head)
        model.freeze_head()
        model.freeze_lang()
        
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
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
        n_batches = len(data_loader)

        if args.save is not None:
            os.makedirs(ckpdir, exist_ok=True)
        
        model = model.cuda()
        model.train()
        if split_idx > 0:
            old_model = old_model.cuda()

        for epoch in range(args.epochs):
            for i, batch in enumerate(data_loader):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')
                data_time = time.time() - start_time
                
                if split_idx > 0:
                    old_logits = old_model(inputs)
                    # workaround of DataParallel problem described here:
                    # https://github.com/pytorch/pytorch/issues/31460#issuecomment-909752233
                    new_logits_new_classes, new_features = model(**dict(inputs=inputs, return_features=True))
                    new_logits_old_classes = old_model.classification_head(new_features)
                    clsf_loss = loss_fn(new_logits_new_classes, labels)
                    distill_loss = cross_entropy(new_logits_old_classes, old_logits, exp=0.5)
                    loss = clsf_loss + args.lwf_lamb * distill_loss
                else:
                    logits = model(inputs)
                    loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if step % print_every == 0 or i + 1 == n_batches:
                    percent_complete = 100 * i / len(data_loader)
                    if split_idx == 0:
                        print(
                            f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                            f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                        )
                    else:
                        print(
                            f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                            f"Loss: {loss.item():.6f}\t Loss clsf: {clsf_loss.item():.6f}\tLoss LWF: {distill_loss.item():.6f}\t"
                            f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                        )
        # Evaluate
        # evaluate(image_encoder, args)
        image_encoder = model.module.image_encoder

        if args.save is not None:
            image_encoder.save(ft_path)
            
        if split_idx > 0:
            # aggregate classification head so it encompasses all the previous tasks
            aggr_classification_head = concat_classification_heads([old_model.classification_head, model.module.classification_head])
        
        old_model = deepcopy(model.module)
        if split_idx > 0:
            old_model.classification_head = aggr_classification_head
        old_model.freeze()

    evaluate(image_encoder, args)


if __name__ == '__main__':
    args = parse_arguments()
    
    # args.model = 'ViT-B-16'
    args.lr = 1e-5
    args.batch_size = 128
    args.sequential_finetuning = True
    args.split_strategy = 'class'
    args.save = f'checkpoints/{args.model}/lwf'
    args.eval_datasets = [args.dataset]

    print('='*100)
    print(f'Finetuning {args.model} on {args.dataset} ({args.n_splits} splits)')
    print('='*100)

    finetune(args)
