import timm

def get_original_vit_image_encoder_transforms(mode):
    # the following data_config is given by OriginalViTImageEncoder
    data_config = {'input_size': (3, 224, 224),
                   'interpolation': 'bicubic',
                   'mean': (0.485, 0.456, 0.406),
                   'std': (0.229, 0.224, 0.225),
                   'crop_pct': 0.875,
                   'crop_mode': 'center'}

    if mode == "train":
        return timm.data.create_transform(**data_config, is_training=True)
    else:
        return timm.data.create_transform(**data_config, is_training=False)

