# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import torchvision
from torchvision import transforms as transforms
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory
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
severity = 5
corruption = "gaussian_noise"
# dataset = torchvision.datasets.ImageFolder(root=f'/home/ykojima/dataset/imagenetv2-c/original', transform=transform('valid'))
dataset = torchvision.datasets.ImageFolder(root=f'/home/ykojima/dataset/imagenetv2-c/{corruption}/{severity}', transform=transform('valid'))
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
    with torch.no_grad():
        predicted_img, mask = model.mae(images)
    masked_predicted_img = predicted_img.cpu()[0] * mask.cpu()[0]
    masked_predicted_img = inverse(masked_predicted_img)

    masked_input = images.cpu()[0] * (1 - mask.cpu()[0])
    
    return masked_predicted_img, masked_input


# %%
def one_sample_ttt(model, optimizer, images, device, mask_ratio=0.75):
    images = images.to(device)
    predicted_img, mask = model(images)
    mae_loss = torch.mean((predicted_img - images) ** 2 * mask) / mask_ratio
    mae_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return mae_loss.item()


# %%
from evaluator import imagenet_config

def zeroshot_classifier(model, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                # 80 patterns per class
                texts = [template.format(classname) for template in templates] #format with class
                max_length = 15
                tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
                batch = {key: values.to(device) for key, values in tokens.items()}
                class_embeddings = model.text_encode(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # the norm shape is torch.Size([80, 1])
                class_embedding = class_embeddings.mean(dim=0) # the mean shape is torch.Size([256])
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


# %%
model.load_state_dict(status['model'])
zeroshot_weights = zeroshot_classifier(model.clip, imagenet_config.imagenet_classes, imagenet_config.imagenet_templates)
logit_scale = 100 # this is from open clip repo

def get_score(model, target_image, target):
    with torch.no_grad():
        image_features = model.clip.image_encode(target_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        scores = (logit_scale * image_features @ zeroshot_weights).softmax(dim=-1)
    return scores[0][target].item()


# %%
# parameters
lr = 5e-3
eps =  1e-8
weight_decay = 0.01
mask_ratio=0.75

# target image
target_image, target = dataset[3880]
target_image = target_image.unsqueeze(0)
display(inverse(target_image[0]))

# just for visualization
model.load_state_dict(status['model'])
model.eval()

masked_predicted_img, masked_input = show_reconstructed_image(model, target_image, device)
display(masked_predicted_img)
display(inverse(masked_input))

target_image = target_image.to(device)
with torch.no_grad():
    predicted_img, mask = model.mae(target_image)
mae_loss = torch.mean((predicted_img - target_image) ** 2 * mask) / mask_ratio
initial_loss = mae_loss.item()

initial_score = get_score(model, target_image, target)

# %% [markdown]
# ### Train with single data

# %%
# initialization
model.load_state_dict(status['model'])

optimizer = torch.optim.AdamW(model.mae.parameters(),
            eps=eps, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.mae.parameters(), lr=lr, weight_decay=weight_decay)


epochs = [-1]
losses = [initial_loss]
scores = [initial_score]
predicted_images = [masked_predicted_img]

for epoch in tqdm(range(0, 500)):
    model.train()
    loss = one_sample_ttt(model.mae, optimizer, target_image, device)
    epochs.append(epoch)
    losses.append(loss)
    model.eval()
    score = get_score(model, target_image, target)
    scores.append(score)
    masked_predicted_img, _ = show_reconstructed_image(model, target_image, device)
    predicted_images.append(masked_predicted_img)
    
result_df = pd.DataFrame({'epoch': epochs, 'loss': losses, 'score': scores, 'predicted_image': predicted_images})


# %%
result_df.plot(x='epoch')
display(result_df['predicted_image'][0])
display(result_df['predicted_image'][100])
display(result_df['predicted_image'][499])

# %%
# temporal
fig, ax = plt.subplots()
original_df.plot(x='epoch', y='loss', ax=ax, label='original')
corruption_df.plot(x='epoch', y='loss', ax=ax, label=f'{corruption}, {severity}')
ax.hlines(0.2965, xmin=0, xmax=500, color='black', linestyle='dashed', lw=1, label='0.2965')
ax.hlines(0.7657, xmin=0, xmax=500, color='black', linestyle='dashed', lw=1, label='0.7657')
ax.set_ylabel('Reconstruction Loss')
plt.legend()
plt.title(f'One Sample TTT (SGD, LR={lr}, WD={weight_decay})')

# %% [markdown]
# ### train with all the data
# observe the transition of reconstraction when the model is trained on all corruption data.

# %%
# initialization
model.load_state_dict(status['model'])

optimizer = torch.optim.AdamW(model.mae.parameters(),
            eps=eps, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.mae.parameters(), lr=lr, weight_decay=weight_decay)

tttrainer = TestTimeTrainer(data_loader, optimizer, mask_ratio, device)

epochs = []
losses = []
predicted_images = []

for epoch in range(0, 1):
    model.train()
    # loss= tttrainer(model.mae)
    epochs.append(epoch)
    losses.append(loss)
    model.eval()
    masked_predicted_img, _ = show_reconstructed_image(model, target_image, device)
    predicted_images.append(masked_predicted_img)
    
result_df = pd.DataFrame({'epoch': epochs, 'loss': losses, 'predicted_images': predicted_images})

# %%
