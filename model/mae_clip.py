import torch
from torch import nn

from model.mae import MAE_ViT
from model.clip import CLIP
from model.open_clip import OpenCLIP

class MAECLIPInterface(nn.Module):
    def __init__(self, image_encoder, text_encoder, image_decoder, image_projector, text_projector,
                 temperature=0.07, mask_ratio=0.75, alpha=1.0):
        super().__init__()
        self.clip = CLIP(image_encoder, text_encoder, image_projector, text_projector, temperature)
        self.mae = MAE_ViT(image_encoder, image_decoder, mask_ratio)
        self._mask_ratio = mask_ratio
        self._alpha = alpha

    def forward(self, batch):
        clip_loss, logit_scale = self.clip(batch)
        predicted_img, mask = self.mae(batch['image'])
        mae_loss = torch.mean((predicted_img - batch['image']) ** 2 * mask) / self._mask_ratio

        total_loss = clip_loss + self._alpha * mae_loss
        return total_loss, clip_loss, mae_loss, predicted_img, logit_scale

class PretrainedOpenMAECLIP(nn.Module):
    def __init__(self, image_encoder, image_decoder, image_projector, open_clip,
                 temperature=0.07, mask_ratio=0.75, alpha=1.0):
        super().__init__()
        self.image_encoder = image_encoder
        self.clip = OpenCLIP(image_encoder, image_projector, open_clip, temperature)
        self.mae = MAE_ViT(image_encoder, image_decoder, mask_ratio)
        self._mask_ratio = mask_ratio
        self._alpha = alpha

    def forward(self, batch):
        clip_loss, logit_scale = self.clip(batch)
        predicted_img, mask = self.mae(batch['image'])
        mae_loss = torch.mean((predicted_img - batch['image']) ** 2 * mask) / self._mask_ratio

        total_loss = clip_loss + self._alpha * mae_loss
        return total_loss, clip_loss, mae_loss, predicted_img, logit_scale

