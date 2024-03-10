# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
# ---

# %%
import torch
from torchvision import transforms as transforms
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../')

from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory
from evaluator.evaluator import ZeroShotImageNetEvaluator
from ttt.simple_ttt import TestTimeTrainer

# %%
from misc.config import load_config
from omegaconf import OmegaConf

config = '../config/ttt_mae.yaml'
cfg = load_config(config)
OmegaConf.set_struct(cfg, True)

model_path = '../output/20240304_decoder_layer8_finetune_datacomp_l_s1b-b8k/checkpoint.pth'

device = 'cuda'
mask_ratio = 0.75

# %%
factory = PretrainedOpenCLIPFactory(cfg.model)
model, tokenizer, transform = factory.create()
model = model.to(device)

status = torch.load(model_path, map_location="cuda")

model.load_state_dict(status['model'])
model.eval()

# %%
inverse = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                                transforms.Normalize(mean = [-0.48145466, -0.4578275, -0.40821073],
                                                     std = [ 1., 1., 1. ]),
                                transforms.ToPILImage() # transform tensor to pillow for simple visualization
                               ])

# %%
from imagenetv2_pytorch import ImageNetV2Dataset
from misc.transforms import Corruption, get_corruption_transform

severity = 1
corruption = "gaussian_noise"
transform = get_corruption_transform(Corruption(severity=severity, corruption_name=corruption))
dataset = ImageNetV2Dataset(transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False)

# %%
from PIL import Image

def concat_h(img1, img2, img3, color="white", spacer=10):
    dst = Image.new(
        "RGB", (img1.width + spacer + img2.width + spacer + img3.width, max(img1.height, img2.height, img3.height)), color
    )
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width + spacer, 0))
    dst.paste(img3, (img1.width + spacer + img2.width + spacer, 0))

    return dst

for images, targets in data_loader:
    images = images.to(device)
    predicted_img, mask = model.mae(images)
    mae_loss = torch.mean((predicted_img - images) ** 2 * mask) / mask_ratio
    print(mae_loss)

    # tensor, (3, 224, 224)
    input = images.cpu()[0]
    predicted_img = predicted_img.cpu()[0]
    mask = mask.cpu()[0]

    masked_input = input * (1 - mask)
    masked_predicted_img = predicted_img * mask

    
    input = inverse(input)
    masked_input = inverse(masked_input)
    masked_predicted_img = inverse(masked_predicted_img)

    display(concat_h(input, masked_input, masked_predicted_img))
    break


# %%
def show_reconstructed_image(model, images, device):
    images = images.to(device)
    predicted_img, mask = model.mae(images)
    masked_predicted_img = predicted_img.cpu()[0] * mask.cpu()[0]
    masked_predicted_img = inverse(masked_predicted_img)
    return masked_predicted_img


# %%
def one_sample_ttt(model, optimizer, images, device, mask_ratio=0.75):
    images = images.to(device)
    predicted_img, mask = model(images)
    mae_loss = torch.mean((predicted_img - images) ** 2 * mask) / mask_ratio
    mae_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return mae_loss


# %%
model.load_state_dict(status['model'])
target_image = dataset[100][0].unsqueeze(0)

lr = 5e-4
eps =  1e-8
weight_decay = 0.2
optimizer = torch.optim.AdamW(model.mae.parameters(),
            eps=eps, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.mae.parameters(), lr=lr, weight_decay=weight_decay)

epochs = []
losses = []
predicted_images = []

for epoch in tqdm(range(0, 500)):
    model.train()
    loss = one_sample_ttt(model.mae, optimizer, target_image, device)
    epochs.append(epoch)
    losses.append(loss.item())
    model.eval()
    masked_predicted_img = show_reconstructed_image(model, target_image, device)
    predicted_images.append(masked_predicted_img)
    
    result_df = pd.DataFrame({'epoch': epochs, 'loss': losses, 'predicted_image': predicted_images})
    

# %%
result_df.plot(x='epoch')
display(result_df['predicted_image'][0])
display(result_df['predicted_image'][100])
display(result_df['predicted_image'][499])

# %% [markdown]
# ### train all the data

# %%
model.load_state_dict(status['model'])

lr = 5e-3
eps =  1e-8
weight_decay = 0.2
mask_ratio = 0.2
optimizer = torch.optim.AdamW(model.mae.parameters(),
            eps=eps, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.mae.parameters(), lr=lr, weight_decay=weight_decay)

tttrainer = TestTimeTrainer(data_loader, optimizer, mask_ratio, device)

epochs = []
losses = []
predicted_images = []

for epoch in range(0, 1):
    model.train()
    loss= tttrainer(model.mae)
    epochs.append(epoch)
    losses.append(loss)
    model.eval()
    # masked_predicted_img = show_reconstructed_image(model, target_image, device)
    # predicted_images.append(masked_predicted_img)
    
    result_df = pd.DataFrame({'epoch': epochs, 'loss': losses})

# %%
