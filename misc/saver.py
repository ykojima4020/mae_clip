
import torch

def save_checkpoint(checkpoint_path, model, epoch):
    to_save = {
        'model': model.state_dict(),
        'ecpoh': epoch 
    }
    torch.save(to_save, checkpoint_path)