import torch
from torch import nn

from model.mae import MAE_Encoder, MAE_Decoder, MAE_ViT

from model.clip import CLIP
from model.modules import TextEncoder, ProjectionHead

class MAECLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self._cfg = cfg
        self._mask_ratio = cfg.mask_ratio

        self._image_size = cfg.image_size
        self._patch_size = cfg.patch_size
        self._emb_dim = cfg.image_embedding

        self._encoder_layer = 12 
        self._encoder_head = 4 
        self._decoder_layer = 4 
        self._decoder_head = 4  

        self._alpha = cfg.alpha

        self.image_encoder = MAE_Encoder(self._image_size,
                                    self._patch_size,
                                    self._emb_dim,
                                    self._encoder_layer,
                                    self._encoder_head)
 
        text_encoder = TextEncoder(self._cfg.text_encoder_name, pretrained=True, trainable=False)
        image_projection = ProjectionHead(embedding_dim=self._cfg.image_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)
        text_projection = ProjectionHead(embedding_dim=self._cfg.text_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)

        self.clip = CLIP(self.image_encoder, text_encoder, image_projection, text_projection, self._cfg.temperature)

        self.mae_decoder = MAE_Decoder(self._image_size,
                                  self._patch_size,
                                  self._emb_dim,
                                  self._decoder_layer,
                                  self._decoder_head)
        self.mae = MAE_ViT(self.image_encoder, self.mae_decoder, self._mask_ratio)


    def forward(self, batch):
        clip_loss = self.clip(batch)
        predicted_img, mask = self.mae(batch['image'])
        mae_loss = torch.mean((predicted_img - batch['image']) ** 2 * mask) / self._mask_ratio

        total_loss = clip_loss + self._alpha * mae_loss
        return total_loss, clip_loss, mae_loss, predicted_img

