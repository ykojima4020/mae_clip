import argparse

import torch
import sys
from tqdm import tqdm
import pandas as pd

sys.path.append('../')
from evaluator.evaluator import ZeroShotImageNetEvaluator
from transformers import DistilBertTokenizer
from factory import MAECLIPFactory
from misc.config import load_config
from omegaconf import OmegaConf

from imagenetv2_pytorch import ImageNetV2Dataset
from misc.transforms import Corruption, get_original_vit_image_encoder_transforms, get_corruption_transform


corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
               'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
               'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation on ImageNet-C', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to a pth file')
    parser.add_argument('--output', type=str, default='./output.csv', help='path to a output csv file')
    return parser


class TestTimeTrainer():

    def __init__(self, data_loader, optimizer, device):
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._device = device

        self._mask_ratio = 0.75

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


def main():
    args = get_args_parser()
    args = args.parse_args()

    batch_size = 64
    num_workers = 4
    pin_memory = True
    lr = 1e-3 * batch_size / 256
    eps = 1e-8 
    weight_decay = 0.05 
    device = 'cuda'
    total_epoch = 1

    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)    

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    factory = MAECLIPFactory(cfg.model)
    status = torch.load(args.checkpoint, map_location="cuda")

    severities = []
    corruptions = []
    acc_1 = []
    acc_5 = []
    ttt_acc_1 = []
    ttt_acc_5 = []

    for severity in range(1,6):
        for corruption in corruptions_name:
 
            severities.append(severity)
            corruptions.append(corruption)

            model = factory.create().to("cuda")
            model.load_state_dict(status['model'])
        
            # [NOTE]: freze CLIP parameters
            for name, param in model.named_parameters():
                if ('mae_decoder' in name) or ('image_encoder' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
            transform = get_corruption_transform(Corruption(severity=severity, corruption_name=corruption))
            evaluator = ZeroShotImageNetEvaluator(tokenizer, device, transform)
            eval_res = evaluator(model.clip)
            print('before test time training')
            print(eval_res)
            acc_1.append(eval_res['eval']['imagenet']['top1'])
            acc_5.append(eval_res['eval']['imagenet']['top5'])
        
            train_dataset = ImageNetV2Dataset(transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        
            # optimizer, but no scheduler 
            optimizer = torch.optim.AdamW(model.mae.parameters(),
                        eps=eps, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        
            tttrainer = TestTimeTrainer(train_loader, optimizer, device)
        
            for epoch in range(0, total_epoch):
                tttrainer(model.mae)
        
            eval_res = evaluator(model.clip)
            print('after test time training')
            print(eval_res)
            ttt_acc_1.append(eval_res['eval']['imagenet']['top1'])
            ttt_acc_5.append(eval_res['eval']['imagenet']['top5'])

    df = pd.DataFrame({'severity': severities,
                       'corruption': corruptions,
                       'top1': acc_1,
                       'top5': acc_5,
                       'ttt_top1': ttt_acc_1,
                       'ttt_top5': ttt_acc_5, })
    print(df) 
    df.to_csv(args.output, index=False)   


if __name__ == "__main__":
    main()

