import torch
from torch import nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, image_projection, text_projection, temperature=0.07):
        super().__init__()
        self._image_encoder = image_encoder
        self._text_encoder = text_encoder
        self._image_projection = image_projection
        self._text_projection = text_projection
        self._temperature = 0.07
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self._temperature))
        self.logit_scale = nn.Parameter(torch.ones([]) *  self._temperature)

    def image_encode(self, image):
        # Getting Image and Text Features
        image_features = self._image_encoder(image)[0][0, :, :]
        # print("image_features: ", image_features.shape)
        image_embeddings = self._image_projection(image_features)
        return image_embeddings 

    def text_encode(self, input_ids, attention_mask):
        text_features = self._text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self._text_projection(text_features)
        return text_embeddings

    def loss(self, image_embeddings, text_embeddings):
        # Calculating the Loss
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits = (text_embeddings @ image_embeddings.T) * logit_scale

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            ((images_similarity + texts_similarity) / 2) * logit_scale, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean(), logit_scale

    def forward(self, batch):
        image_embeddings = self.image_encode(batch['image'])
        text_embeddings = self.text_encode(batch['input_ids'], batch['attention_mask'])

        # Calculating the Loss
        # collect embeddings from all GPUs for calculating a contrastive loss
        image_embeddings_gather = concat_all_gather(image_embeddings)
        text_embeddings_gather = concat_all_gather(text_embeddings)

        loss = self.loss(image_embeddings, text_embeddings)
        return loss

    def get_transforms(self, mode):
        return self._image_encoder.get_transforms(mode)

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# utils
# this is a function from https://github.com/facebookresearch/moco/blob/main/moco/builder.py#L178
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # clip = CLIP()
    # loss = clip(batch)
