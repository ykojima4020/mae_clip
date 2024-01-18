from model.clip import CLIP
from model.modules import ImageEncoder, ViTImageEncoder, OriginalViTImageEncoder, TextEncoder, ProjectionHead

from model.mae_clip import MAECLIP

class Factory:
    def __init__(self, cfg):
        pass

    def create(self):
        raise NotImplementedError

class ViTCLIPFactory(Factory):

    def __init__(self, cfg):
        self._cfg = cfg

    def create(self):
        image_encoder = ViTImageEncoder(self._cfg.image_encoder_name, self._cfg.image_encoder_pretrained, self._cfg.image_encoder_trainable)
        text_encoder = TextEncoder(self._cfg.text_encoder_name, self._cfg.text_encoder_pretrained, self._cfg.text_encoder_trainable)
        image_projection = ProjectionHead(embedding_dim=self._cfg.image_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)
        text_projection = ProjectionHead(embedding_dim=self._cfg.text_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)

        return CLIP(image_encoder, text_encoder, image_projection, text_projection, self._cfg.temperature)

class OriginalViTCLIPFactory(Factory):

    def __init__(self, cfg):
        self._cfg = cfg

    def create(self):
        image_encoder = OriginalViTImageEncoder(self._cfg.image_encoder_name, self._cfg.image_encoder_pretrained, self._cfg.image_encoder_trainable, self._cfg)
        text_encoder = TextEncoder(self._cfg.text_encoder_name, pretrained=True, trainable=False)
        image_projection = ProjectionHead(embedding_dim=self._cfg.image_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)
        text_projection = ProjectionHead(embedding_dim=self._cfg.text_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)

        return CLIP(image_encoder, text_encoder, image_projection, text_projection, self._cfg.temperature)

class MAECLIPFactory(Factory):

    def __init__(self, cfg):
        self._cfg = cfg

    def create(self):
        return MAECLIP(self._cfg)

