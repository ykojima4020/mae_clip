
import torch

def save_checkpoint(checkpoint_path, model, epoch, logit_scale):
    to_save = {
        'model': model.state_dict(),
        'ecpoh': epoch,
        'logit_scale': logit_scale 
    }
    torch.save(to_save, checkpoint_path)