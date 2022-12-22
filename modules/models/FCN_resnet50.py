from torchvision import models
import torch.nn as nn

class FCN_ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
        
    def forward(self, x):
        return self.model(x)['out']