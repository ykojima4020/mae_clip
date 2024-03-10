import argparse
import torch
import sys
from tqdm import tqdm
import pandas as pd
import wandb

sys.path.append('../')
from evaluator.evaluator import ZeroShotImageNetEvaluator
from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory
from misc.config import load_config
from omegaconf import OmegaConf

from imagenetv2_pytorch import ImageNetV2Dataset
from misc.transforms import Corruption, get_corruption_transform

from misc.utils import AvgMeter

corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
                    'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                    'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation on ImageNet-C', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--type', default='normal', choices=['normal', 'open'], help='a kind of archtectures')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to a pth file')
    parser.add_argument('--output', type=str, default='./output.csv', help='path to a output csv file')
    return parser


class TestTimeTrainer():

    def __init__(self, data_loader, optimizer, mask_ratio, device):
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._device = device
        self._mask_ratio = mask_ratio
        self._mae_loss_meter = AvgMeter()


    def __call__(self, model):
        tqdm_object = tqdm(self._data_loader, total=len(self._data_loader))
        for idx, (images, target) in enumerate(tqdm_object):
            images = images.to(self._device)

            predicted_img, mask = model(images)
            # [NOTE]: loss calculation is included.
            mae_loss = torch.mean((predicted_img - images) ** 2 * mask) / self._mask_ratio
            mae_loss.backward() 
            self._optimizer.step()
            self._optimizer.zero_grad()

            count = images.size(0)
            self._mae_loss_meter.update(mae_loss.item(), count)

        return self._mae_loss_meter.avg

def sweep():
    sweep_config = load_config('ttt_sweep.yaml')
    sweep_id = wandb.sweep(OmegaConf.to_container(sweep_config, resolve=True),
                           project="mae_clip_ttt")
    wandb.agent(sweep_id, main)

def main():
    args = get_args_parser()
    args = args.parse_args()

    # fixed parameters
    eps = 1e-8 
    device = 'cuda'

    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)    

    if args.wandb:
        run = wandb.init(project='mae_clip_ttt',
                         entity="ykojima",
                         config=OmegaConf.to_container(cfg, resolve=True))
        config = wandb.config
        print(config)

    if args.type == 'normal':
        factory = RILSMAECLIPFactory(cfg.model)
    elif args.type == 'open':
        factory = PretrainedOpenCLIPFactory(cfg.model)
    else:
        raise TypeError

    model, tokenizer, _ = factory.create()
    model = model.to(device)
    # [NOTE]: freze CLIP parameters
    for name, param in model.named_parameters():
        if ('image_encoder' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
        # print(name, param.requires_grad)
 
    status = torch.load(args.checkpoint, map_location="cuda")

    severity = 5
    corruption = "gaussian_noise"

    lr = config.lr_scale * 1e-3 * config.batch_size / 256
    optimizer = torch.optim.AdamW(model.mae.parameters(),
                eps=eps, lr=lr, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    top1, top5 = run_ttt(model, tokenizer, status, severity, corruption, optimizer, epochs=config.epochs, batch_size=config.batch_size)

    stats = {'top1': top1, 'top5': top5}
    if args.wandb:
        wandb.log(stats)
        # table.add_data(lr, weight_decay, total_epoch, batch_size, severity, corruption, top1, top5)

    del model
    torch.cuda.empty_cache()

    # table.get_dataframe().to_csv(args.output, index=False)
    # wandb.log({'result': table})

    if args.wandb:
        run.finish()

def run_ttt(model, tokenizer, status, severity, corruption, optimizer,
            epochs=1, batch_size=64, num_workers=4, pin_memory=True,
            mask_ratio=0.75, device='cuda', ttt=True):
    model.load_state_dict(status['model'])

    transform = get_corruption_transform(Corruption(severity=severity, corruption_name=corruption))
    evaluator = ZeroShotImageNetEvaluator(tokenizer, device, transform)

    train_dataset = ImageNetV2Dataset(transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    if ttt:
        tttrainer = TestTimeTrainer(train_loader, optimizer, mask_ratio, device)

        for epoch in range(0, epochs):
            tttrainer(model.mae)

    eval_res = evaluator(model.clip)
    top1 = eval_res['eval']['imagenet']['top1']
    top5 = eval_res['eval']['imagenet']['top5']
    return top1, top5 

if __name__ == "__main__":
    main()
    # sweep()
