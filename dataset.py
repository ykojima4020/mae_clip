import os
import cv2
import torch
from torchvision import transforms
import timm

import config as CFG
from PIL import Image

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(f"{CFG.image_path}/{self.image_filenames[idx]}") 
        if image.mode == 'L':
            image = image.convert('RGB') # This is because there's images in a gray scale.
        # image = self.transforms(image=image)['image']
        image = self.transforms(image)
        # item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['image'] = image
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)

