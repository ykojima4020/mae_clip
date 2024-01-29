import torch
import torch.distributed as dist

from data.dataset import CLIPDataset
from misc.transforms import get_original_vit_image_encoder_transforms
from misc.coco_captions_to_df import get_coco_captions_df, get_coco_captions_test_df

class CLIPDataLoaderBuilder():

    def __init__(self, tokenizer, batch_size, num_workers):
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers

    def __call__(self, image_path, annotation_json, mode, rank, world_size, test=False):
        transforms = get_original_vit_image_encoder_transforms(mode)
        if not test:
            dataframe = get_coco_captions_df(annotation_json)
        else:
            dataframe = get_coco_captions_test_df(annotation_json)

        dataset = CLIPDataset(
            image_path,
            dataframe["image"].values,
            dataframe["caption"].values,
            tokenizer=self._tokenizer,
            transforms=transforms,
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=dist.get_rank(),
            shuffle=True if mode == "train" else False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, num_workers=self._num_workers,
            shuffle=sampler is None,
            sampler=sampler)
        return dataloader, sampler
    
    