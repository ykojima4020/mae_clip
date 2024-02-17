import os
import sys
from tqdm import tqdm
import pathlib
import wandb
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from transformers import DistilBertTokenizer
from omegaconf import OmegaConf

from factory import MAECLIPFactory
from data.dataloader_builder import CLIPDataLoaderBuilder, GCC3MDataLoaderBuilder
from trainer.trainer import SimpleTrainer
from trainer.validater import SimpleValidater
from evaluator.evaluator import ZeroShotImageNetEvaluator
from misc.utils import AvgMeter, get_lr
from misc.saver import save_checkpoint
from misc.config import get_config
from misc.lr_scheduler import build_scheduler

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


def main(rank, world_size, cfg):

    print(f"Running train on rank {rank}.")
    setup(rank, world_size)

    output_dir = pathlib.Path(cfg.output)

    if dist.get_rank() == 0:
        print(cfg)

    if not cfg.train.lr:
        cfg.train.lr = cfg.train.base_lr * cfg.data.batch_size * world_size / 256 # 1e-3 * 64 / 256 = 0.00025

    if cfg.wandb and dist.get_rank() == 0:
        run = wandb.init(project="mae_clip", entity="ykojima", config=OmegaConf.to_container(cfg, resolve=True)) 

    tokenizer = DistilBertTokenizer.from_pretrained(cfg.model.text.encoder.name)
    factory = MAECLIPFactory(cfg.model)

    model = factory.create().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataloader_builder = CLIPDataLoaderBuilder(cfg.data, tokenizer)
    gcc3m_dataloader_builder = GCC3MDataLoaderBuilder(cfg.data, tokenizer)

    train_loader, train_sampler = gcc3m_dataloader_builder('train', rank, world_size, test=cfg.test)

    val_loader, _ = dataloader_builder(cfg.data.dataset.val_image_path,
                                      cfg.data.dataset.val_json, 'val', rank, world_size, test=cfg.test)

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), eps=cfg.train.optimizer.eps, betas=cfg.train.optimizer.betas,
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(train_loader))

    trainer = SimpleTrainer(train_loader, optimizer, lr_scheduler, cfg.train.clip_grad, rank)
    validater = SimpleValidater(val_loader, optimizer, rank)
    if dist.get_rank() == 0:
        evaluator = ZeroShotImageNetEvaluator(tokenizer, rank)

    best_loss = float('inf')
    for epoch in range(cfg.train.epochs):
        dist.barrier()
        # [TODO]: when using webdataset, sampler can not be used. How should I do?
        # train_sampler.set_epoch(epoch)
        stats = {'epoch': epoch}
        print(f"Epoch: {epoch + 1}")
        ddp_model.train()
        train_stats = trainer(ddp_model, epoch)
        stats = stats | train_stats
        ddp_model.eval()
        with torch.no_grad():
            valid_stats, table = validater(ddp_model)
            stats = stats | valid_stats

        if stats['valid']['loss'] < best_loss and dist.get_rank() == 0:
            best_loss = stats['valid']['loss']
            checkpoint = output_dir / f"checkpoint_{epoch+1}.pth"
            save_checkpoint(checkpoint, ddp_model.module, epoch, stats['logit_scale'])
            print("Saved Best Model!")
        if dist.get_rank() == 0:
            eval_stats = evaluator(ddp_model.module.clip)
            stats = stats | eval_stats
        if cfg.wandb and dist.get_rank() == 0:
            wandb.log(stats)
            wandb.log({'image': table})

    if cfg.wandb and dist.get_rank() == 0:
        run.finish()

    cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    cfg = get_config(args)
    if cfg.output:
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    mp.spawn(main,
        args=(cfg.world_size, cfg,),
        nprocs=cfg.world_size,
        join=True)
