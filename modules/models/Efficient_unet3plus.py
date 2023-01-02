import torch.nn as nn
import torch
from unet3plus.model import Unet3Plus

class Efficient_UNet_3Plus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # model 불러오기
        # 출력 label 수 정의 (classes=11)
        self.model = Unet3Plus(
            encoder_name="efficientnet-b4", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=11,                     # model output channels (number of classes in your dataset)
            decoder_channels=220,
            skip_channels=64,
        )
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    model = Efficient_UNet_3Plus().to('cuda')
    
    x = torch.rand((1, 3, 256, 256)).to('cuda')
    
    out = model(x)
    print(out.shape)