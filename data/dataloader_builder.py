import os.path as osp

import torch

from data.dataset import CLIPDataset
from misc.transforms import get_original_vit_image_encoder_transforms
from misc.coco_captions_to_df import get_coco_captions_df, get_coco_captions_test_df

from braceexpand import braceexpand
import webdataset as wds

class CLIPDataLoaderBuilder():

    def __init__(self, tokenizer, batch_size, num_workers):
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers

    def __call__(self, image_path, annotation_json, mode, test=False):
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
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, num_workers=self._num_workers,
            shuffle=True if mode == "train" else False)
        return dataloader

class GCC3MDataLoaderBuilder():

    def __init__(self, cfg, tokenizer, batch_size, num_workers):
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._max_length = cfg.text_aug.max_seq_len
        self._cfg = cfg

    def __call__(self, mode, test=False):

        transforms = get_original_vit_image_encoder_transforms(mode)

        dataset_type = None
        tar_file_list = []
        total_length = 0
        for ds in self._cfg.dataset[mode]:
            ds_meta = self._cfg.dataset.meta[ds]
            if dataset_type is None:
                dataset_type = ds_meta.type
            else:
                assert dataset_type == ds_meta.type, \
                    'All datasets must be of the same type'
    
            prefix = ds_meta.prefix
            path = ds_meta.path
            length = ds_meta.length
            cur_tar_file_list = []
            for tar_file in braceexpand(osp.join(path, prefix)):
                if osp.exists(tar_file):
                    cur_tar_file_list.append(tar_file)
            print(f'Found {len(cur_tar_file_list)} files for dataset {ds}')
            tar_file_list.extend(cur_tar_file_list)
            total_length += length
        print(f'Found {len(tar_file_list)} files in total for split {mode}')

        dataset = wds.WebDataset(tar_file_list)
        dataset = dataset.decode('pil')
        dataset = dataset.rename(image='jpg;png;jpeg', text='text;txt', keep=False)

        dataset = dataset.map_dict(
            image=transforms,
            text=self._text_transform
            )
        dataset = dataset.with_length(total_length)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, num_workers=self._num_workers,
            shuffle=False, # invalid shuffle for webdata
            collate_fn=self._collate_cb)
        return dataloader

    def _collate_cb(self, batch):
        images = list()
        input_ids = list()
        attention_masks = list()
    
        for b in batch:
            images.append(b['image'])
            input_ids.append(b['text']['input_ids'])
            attention_masks.append(b['text']['attention_mask'])

        images = torch.stack(images, dim=0)
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_masks = torch.tensor(attention_masks, dtype=torch.int64)

        return {'image': images, 'input_ids': input_ids, 'attention_mask': attention_masks} 

    def _text_transform(self, x):
        # [NOTE]: truncation cut the sequence in a fixed size.
        encode = self._tokenizer(x, padding="max_length", truncation=True, max_length=self._max_length)
        return encode
