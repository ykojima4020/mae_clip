import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib
import wandb

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertTokenizer

import config as CFG
from dataset import CLIPDataset, get_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr

from coco_captions_to_df import get_coco_captions_df, get_coco_captions_test_df

def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    # wandb_config = vars(CFG)
    run = wandb.init(project="mae_clip", entity="ykojima") 

    # train_df, valid_df = make_train_valid_dfs()
    train_df = get_coco_captions_test_df(CFG.train_json)
    valid_df = get_coco_captions_test_df(CFG.val_json)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    print("tokenizer created.")
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    print(CFG.checkpoints)
    save_dir = pathlib.Path(CFG.checkpoints)
    save_dir.mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(CFG.logdir)

    model = CLIPModel().to(CFG.device)
    print("CLIP created.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        stats = {'epoch': epoch}
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        stats['train_loss'] = train_loss.avg
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            stats['val_loss'] = valid_loss.avg 

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            checkpoint = save_dir / f"checkpoint_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint)
            print("Saved Best Model!")
        # writer.add_scalars('loss', {'train' : train_loss.avg, 'val' : valid_loss.avg}, global_step=epoch)
        wandb.log(stats)

    run.finish()

if __name__ == "__main__":
    main()
