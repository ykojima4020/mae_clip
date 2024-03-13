import torch
from tqdm import tqdm
from misc.utils import AvgMeter

class TestTimeTrainer():

    def __init__(self, data_loader, optimizer, mask_ratio, device):
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._device = device
        self._mask_ratio = mask_ratio
        self._mae_loss_meter = AvgMeter()

    def __call__(self, model):
        tqdm_object = tqdm(self._data_loader, total=len(self._data_loader))
        for idx, (images, target) in enumerate(tqdm_object):
            images = images.to(self._device)

            predicted_img, mask = model(images)
            # [NOTE]: loss calculation is included.
            mae_loss = torch.mean((predicted_img - images) ** 2 * mask) / self._mask_ratio
            mae_loss.backward() 
            self._optimizer.step()
            self._optimizer.zero_grad()

            count = images.size(0)
            self._mae_loss_meter.update(mae_loss.item(), count)

        return self._mae_loss_meter.avg
