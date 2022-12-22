import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

class Efficient_UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # model 불러오기
        # 출력 label 수 정의 (classes=11)
        self.model = smp.Unet(
            encoder_name="efficientnet-b0", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=11,                     # model output channels (number of classes in your dataset)
        )
        
    def forward(self, x):
        return self.model(x)