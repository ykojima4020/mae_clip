import torch
import torch.distributed as dist
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

    def loss(self, image_x, text_x):

        batch_size = image_x.shape[0]

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        image_x_all = concat_all_gather(image_x)
        text_x_all = concat_all_gather(text_x)
 
        images_similarity = image_x_all @ image_x_all.T
        texts_similarity = text_x_all @ text_x_all.T
        targets = F.softmax(((images_similarity + texts_similarity) / 2) * logit_scale, dim=-1) # [B, B]

        start = dist.get_rank() * batch_size
        end = start + batch_size
        targets = targets[start:end]             # [B/gpus, B]

        logits_per_img = image_x @ text_x_all.T  # [B/gpus, B]
        logits_per_text = text_x @ image_x_all.T # [B/gpus, B]

        texts_loss = cross_entropy(logits_per_text, targets, reduction='mean')
        images_loss = cross_entropy(logits_per_img, targets, reduction='mean')
        loss =  (images_loss + texts_loss) / 2.0
        return loss, logit_scale

    def forward(self, batch):
        image_x = self.image_encode(batch['image'])
        loss, logit_scale = self.loss(image_x, text_x)

        return loss, logit_scale

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
