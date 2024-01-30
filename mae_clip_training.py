import os
import sys
from tqdm import tqdm
import pathlib
import wandb
import argparse

import torch
from transformers import DistilBertTokenizer
from omegaconf import OmegaConf

from factory import MAECLIPFactory
from data.dataloader_builder import CLIPDataLoaderBuilder
from trainer.trainer import SimpleTrainer
from trainer.validater import SimpleValidater
from misc.utils import AvgMeter, get_lr
from misc.saver import save_checkpoint
from misc.config import get_config

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP pre-training', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--batch_size', type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', type=str)

    # Dataset parameters
    parser.add_argument('--train_image_path', default='./dataset/coco/',
                        help='path to the training images')
    parser.add_argument('--train_json', default='/home/ykojima/dataset/coco/annotations/captions_train2014.json',
                        help='training annotation json')
    parser.add_argument('--val_image_path', default='./dataset/coco/',
                        help='path to the validation images')
    parser.add_argument('--val_json', default='/home/ykojima/dataset/coco/annotations/captions_val2014.json',
                        help='validation annotation json')
    parser.add_argument('--output', default='./tmp',
                        help='path where to save, empty for no saving')
    return parser


def main(cfg):

    device = torch.device(cfg.device)
    output_dir = pathlib.Path(cfg.output)

    print(cfg)
    if not cfg.train.lr:
        cfg.train.lr = cfg.train.base_lr * cfg.data.batch_size / 256 # 1e-3 * 64 / 256 = 0.00025

    if cfg.wandb:
        run = wandb.init(project="mae_clip", entity="ykojima", config=OmegaConf.to_container(cfg, resolve=True)) 

    tokenizer = DistilBertTokenizer.from_pretrained(cfg.model.text.encoder.name)
    factory = MAECLIPFactory(cfg.model)
    model = factory.create().to(device)

    dataloader_builder = CLIPDataLoaderBuilder(tokenizer, cfg.data.batch_size, cfg.data.num_workers)
    train_loader = dataloader_builder(cfg.data.dataset.train_image_path,
                                      cfg.data.dataset.train_json, 'train', test=cfg.test)
    val_loader = dataloader_builder(cfg.data.dataset.val_image_path,
                                      cfg.data.dataset.val_json, 'val', test=cfg.test)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.train.lr_scheduler.patience, factor=cfg.train.lr_scheduler.factor)

    trainer = SimpleTrainer(train_loader, optimizer, device)
    validater = SimpleValidater(train_loader, optimizer, device)

    best_loss = float('inf')
    for epoch in range(cfg.train.epochs):
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
        if cfg.wandb:
            wandb.log(stats)
            wandb.log({'image': table})

    if cfg.wandb:
        run.finish()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    cfg = get_config(args)
    device = torch.device(cfg.device)
    if cfg.output:
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    main(cfg)
