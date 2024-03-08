import open_clip
from transformers import DistilBertTokenizer

from model.clip import CLIP
from model.modules import TextEncoder, ProjectionHead
from model.mae import MAE_Encoder, MAE_Decoder 
from model.mae_clip import PretrainedOpenMAECLIP, MAECLIPInterface
from model.models_rils import RILSMAEEncoder
from model.open_clip import OpenCLIPImageEncoder, OpenCLIPImageProjector
from model.tokenizer import BertTokenizer, OpenCLIPTokenizer

from misc.transforms import get_original_vit_image_encoder_transforms, get_open_clip_vitb16_transforms

class Factory:
    def __init__(self, cfg):
        pass

    def create(self):
        raise NotImplementedError

class OriginalViTCLIPFactory(Factory):

    def __init__(self, cfg):
        # [NOTE]: the following parameters are set by a configuration file.
        self._cfg = cfg
        self._image_size = 224 
        self._patch_size = 14
        self._emb_dim = 512
        self._encoder_layer = 12
        self._encoder_head = 4 
 
    def create(self):
        image_encoder = MAE_Encoder(self._image_size, self._patch_size, self._emb_dim, self._encoder_layer, self._encoder_head)
        text_encoder = TextEncoder(self._cfg.text_encoder_name, pretrained=True, trainable=False)
        image_projection = ProjectionHead(embedding_dim=self._cfg.image_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)
        text_projection = ProjectionHead(embedding_dim=self._cfg.text_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)
        return CLIP(image_encoder, text_encoder, image_projection, text_projection, self._cfg.temperature)

class RILSMAECLIPFactory(Factory):

    def __init__(self, cfg):
        self._cfg = cfg

        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size

        self._emb_dim = cfg.image.encoder.embeddings
        self._encoder_layer = cfg.image.encoder.layer
        self._encoder_head = cfg.image.encoder.head 
        self._decoder_layer = cfg.image.decoder.layer 
        self._decoder_head = cfg.image.decoder.head 

        self._alpha = cfg.loss.alpha
        self._mask_ratio = cfg.mae.mask_ratio
        self._temperature = cfg.clip.temperature

    def create(self):
        image_encoder = RILSMAEEncoder(
            img_size=self._image_size, patch_size=self._patch_size,
            embed_dim=self._emb_dim, depth=self._encoder_layer,
            num_heads=self._encoder_head)

        image_decoder = MAE_Decoder(
            image_size=self._image_size, patch_size=self._patch_size,
            emb_dim=self._emb_dim, num_layer=self._decoder_layer,
            num_head=self._decoder_head)
 
        text_encoder = TextEncoder(self._cfg.text.encoder.name, pretrained=self._cfg.text.encoder.pretrained, trainable=self._cfg.text.encoder.trainable)
        image_projector = ProjectionHead(embedding_dim=self._emb_dim, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)
        text_projector = ProjectionHead(embedding_dim=self._cfg.text.encoder.embeddings, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)

        model = MAECLIPInterface(image_encoder, text_encoder, image_decoder,
                image_projector, text_projector, self._temperature, self._mask_ratio, self._alpha)

        tokenizer = DistilBertTokenizer.from_pretrained(self._cfg.text.encoder.name)
        tokenizer = BertTokenizer(tokenizer)

        transform = get_original_vit_image_encoder_transforms

        return model, tokenizer, transform


class PretrainedOpenCLIPFactory(Factory):
    def __init__(self, cfg):
        self._cfg = cfg

        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size

        self._emb_dim = cfg.image.encoder.embeddings
        self._encoder_layer = cfg.image.encoder.layer
        self._encoder_head = cfg.image.encoder.head 
        self._decoder_layer = cfg.image.decoder.layer 
        self._decoder_head = cfg.image.decoder.head 

        self._alpha = cfg.loss.alpha
        self._mask_ratio = cfg.mae.mask_ratio
        self._temperature = cfg.clip.temperature

    def create(self):

        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b-b8k')

        image_encoder = OpenCLIPImageEncoder(model)
        image_projector = OpenCLIPImageProjector(model)

        image_decoder = MAE_Decoder(
            image_size=self._image_size, patch_size=self._patch_size,
            emb_dim=self._emb_dim, num_layer=self._decoder_layer,
            num_head=self._decoder_head)

        model = PretrainedOpenMAECLIP(image_encoder, image_decoder,
                image_projector, model, self._temperature, self._mask_ratio, self._alpha)

        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        tokenizer = OpenCLIPTokenizer(tokenizer)

        transform = get_open_clip_vitb16_transforms

        return model, tokenizer, transform
 
