import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
import albumentations as A

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

    def get_transforms(self, mode):
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(CFG.size, CFG.size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(CFG.size, CFG.size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ]
            )
    

class ViTImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

    def get_transforms(self, mode):
        data_config = timm.data.resolve_model_data_config(self.model)

        if mode == "train":
            return timm.data.create_transform(**data_config, is_training=True)
        else:
            return timm.data.create_transform(**data_config, is_training=False)



class TextEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True, trainable=False):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

