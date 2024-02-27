import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import diffdist.functional as diff_dist

import open_clip

class OpenCLIP(nn.Module):
    def __init__(self, image_encoder=None, text_encoder=None, image_projection=None, text_projection=None, temperature=0.07):
        super().__init__()
        self._temperature = 0.07
        self.logit_scale = nn.Parameter(torch.ones([]) *  self._temperature)
        self.cross_entropy = nn.CrossEntropyLoss()
        self._model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        print('open_clip model loaded.')

    def loss(self, image_x, text_x):
        batch_size = image_x.shape[0]
        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()

        # [B, C]
        image_x = F.normalize(image_x, dim=-1)
        text_x = F.normalize(text_x, dim=-1)

        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)
        return loss, logit_scale

    def image_encode(self, image):
        # Getting Image and Text Features
        image_x = self._model.encode_image(image)
        return image_x 

    def text_encode(self, text):
        text_x = self._model.encode_text(text)
        return text_x

    def forward(self, batch):
        image_x = self.image_encode(batch['image'])
        text_x = self.text_encode(batch['text'])
        loss, logit_scale = self.loss(image_x, text_x)

        return loss, logit_scale

#[NOTE]: https://github.com/NVlabs/GroupViT/blob/main/models/multi_label_contrastive.py#L24C1-L34C51
def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


