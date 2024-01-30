import os
import sys
from tqdm import tqdm
import pathlib
import wandb
import argparse

import torch
from transformers import DistilBertTokenizer

from factory import MAECLIPFactory
from data.dataloader_builder import CLIPDataLoaderBuilder
from trainer.trainer import SimpleTrainer
from trainer.validater import SimpleValidater
from misc.utils import AvgMeter, get_lr
from misc.saver import save_checkpoint

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP pre-training', add_help=False)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--num_workers', default=4, type=int, help='number of processes loading data')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Shared training
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--alpha', default=1, type=float,
                        help='weight of mae reconstruction loss')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--break_after_epoch', type=int, metavar='N', 
                        help='break training after X epochs, to tune hyperparams and avoid messing with training schedule.')

    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--factor', type=float, default=0.8)
    

    # Tokenizer parameters
    parser.add_argument('--text_tokenizer', default='distilbert-base-uncased',
                        help='text_tokenizer')

    # Model parameters
    parser.add_argument('--text_encoder_name', default='distilbert-base-uncased')
    parser.add_argument('--text_encoder_pretrained', action='store_true')
    parser.add_argument('--text_encoder_trainable', action='store_true')

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=14)

    parser.add_argument('--image_encoder_name', default='vit_base_patch16_224.augreg2_in21k_ft_in1k')
    parser.add_argument('--image_encoder_pretrained', action='store_true')
    parser.add_argument('--image_encoder_trainable', action='store_true')

    # Projection parameters
    parser.add_argument('--text_embedding', type=int, default=768)
    parser.add_argument('--image_embedding', type=int, default=512)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--temperature', type=float, default=1.0)


    # Dataset parameters
    parser.add_argument('--train_image_path', default='./dataset/coco/',
                        help='path to the training images')
    parser.add_argument('--train_json', default='/home/ykojima/dataset/coco/annotations/captions_train2014.json',
                        help='training annotation json')
    parser.add_argument('--val_image_path', default='./dataset/coco/',
                        help='path to the validation images')
    parser.add_argument('--val_json', default='/home/ykojima/dataset/coco/annotations/captions_val2014.json',
                        help='validation annotation json')
    parser.add_argument('--output_dir', default='./tmp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    return parser


def main(args):

    device = torch.device(args.device)
    output_dir = pathlib.Path(args.output_dir)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256 # 1e-3 * 64 / 256 = 0.00025

    if args.wandb:
        wandb_config = vars(args)
        run = wandb.init(project="mae_clip", entity="ykojima", config=wandb_config) 

    tokenizer = DistilBertTokenizer.from_pretrained(args.text_tokenizer)
    factory = MAECLIPFactory(args)
    model = factory.create().to(device)

    dataloader_builder = CLIPDataLoaderBuilder(tokenizer, args.batch_size, args.num_workers)
    train_loader = dataloader_builder(args.train_image_path,
                                      args.train_json, 'train', test=args.test)
    val_loader = dataloader_builder(args.val_image_path,
                                      args.val_json, 'val', test=args.test)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience, factor=args.factor)

    trainer = SimpleTrainer(train_loader, optimizer, device)
    validater = SimpleValidater(train_loader, optimizer, device)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        stats = {'epoch': epoch}
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_stats = trainer(model)
        stats = stats | train_stats
        model.eval()
        with torch.no_grad():
            # valid_stats, table = valid_epoch(model, val_loader)
            valid_stats, table = validater(model)
            stats = stats | valid_stats

        if stats['valid']['loss'] < best_loss:
            best_loss = stats['valid']['loss']
            checkpoint = output_dir / f"checkpoint_{epoch+1}.pth"
            save_checkpoint(checkpoint, model, epoch)
            print("Saved Best Model!")
        lr_scheduler.step(stats['valid']['loss'])
        if args.wandb:
            wandb.log(stats)
            wandb.log({'image': table})

    if args.wandb:
        run.finish()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    device = torch.device(args.device)
    if args.output_dir:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
