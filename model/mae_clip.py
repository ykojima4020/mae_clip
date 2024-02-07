import torch
from torch import nn

from model.mae import MAE_Encoder, MAE_Decoder, MAE_ViT

from model.clip import CLIP
from model.modules import TextEncoder, ProjectionHead

from model.models_rils import RILSMAEEncoder, RILSMAEDecoder

class MAECLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self._cfg = cfg
        self._mask_ratio = cfg.mae.mask_ratio

        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size
        self._emb_dim = cfg.image.encoder.embeddings

        self._encoder_layer = 12 
        self._encoder_head = 4 
        self._decoder_layer = 4 
        self._decoder_head = 4  

        self._alpha = cfg.loss.alpha

        self.image_encoder = MAE_Encoder(self._image_size,
                                    self._patch_size,
                                    self._emb_dim,
                                    self._encoder_layer,
                                    self._encoder_head)

        text_encoder = TextEncoder(self._cfg.text.encoder.name, pretrained=self._cfg.text.encoder.pretrained, trainable=self._cfg.text.encoder.trainable)
        image_projection = ProjectionHead(embedding_dim=self._emb_dim, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)
        text_projection = ProjectionHead(embedding_dim=self._cfg.text.encoder.embeddings, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)

        self.clip = CLIP(self.image_encoder, text_encoder, image_projection, text_projection, self._cfg.clip.temperature)

        self.mae_decoder = MAE_Decoder(self._image_size,
                                  self._patch_size,
                                  self._emb_dim,
                                  self._decoder_layer,
                                  self._decoder_head)
        self.mae = MAE_ViT(self.image_encoder, self.mae_decoder, self._mask_ratio)


    def forward(self, batch):
        clip_loss, logit_scale = self.clip(batch)
        predicted_img, mask = self.mae(batch['image'])
        mae_loss = torch.mean((predicted_img - batch['image']) ** 2 * mask) / self._mask_ratio

        total_loss = clip_loss + self._alpha * mae_loss
        return total_loss, clip_loss, mae_loss, predicted_img, logit_scale


class RILSMAECLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self._cfg = cfg
        self._mask_ratio = cfg.mae.mask_ratio

        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size

        self._emb_dim = cfg.image.encoder.embeddings
        self._encoder_layer = cfg.image.encoder.layer
        self._encoder_head = cfg.image.encoder.head 
        self._decoder_layer = cfg.image.decoder.layer 
        self._decoder_head = cfg.image.decoder.head 

        self._alpha = cfg.loss.alpha

        self.image_encoder = RILSMAEEncoder(
            img_size=self._image_size, patch_size=self._patch_size,
            embed_dim=self._emb_dim, depth=self._encoder_layer,
            num_heads=self._encoder_head)

        self.mae_decoder = MAE_Decoder(
            image_size=self._image_size, patch_size=self._patch_size,
            emb_dim=self._emb_dim, num_layer=self._decoder_layer,
            num_head=self._decoder_head)
 
        text_encoder = TextEncoder(self._cfg.text.encoder.name, pretrained=self._cfg.text.encoder.pretrained, trainable=self._cfg.text.encoder.trainable)
        image_projection = ProjectionHead(embedding_dim=self._emb_dim, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)
        text_projection = ProjectionHead(embedding_dim=self._cfg.text.encoder.embeddings, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)

        self.clip = CLIP(self.image_encoder, text_encoder, image_projection, text_projection, self._cfg.clip.temperature)

        self.mae = MAE_ViT(self.image_encoder, self.mae_decoder, self._mask_ratio)

    def forward(self, batch):
        clip_loss, logit_scale = self.clip(batch)
        predicted_img, mask = self.mae(batch['image'])
        mae_loss = torch.mean((predicted_img - batch['image']) ** 2 * mask) / self._mask_ratio

        total_loss = clip_loss + self._alpha * mae_loss
        return total_loss, clip_loss, mae_loss, predicted_img, logit_scale

