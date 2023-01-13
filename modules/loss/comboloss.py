import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()
        self.diceloss = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        self.ce = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)

    def forward(self, inputs, targets):
        combo = 0.5 * self.diceloss(inputs, targets) + 0.5 * self.ce(inputs, targets)
        return combo
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.diceloss = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)

    def forward(self, inputs, targets):
        loss = self.diceloss(inputs, targets)
        return loss