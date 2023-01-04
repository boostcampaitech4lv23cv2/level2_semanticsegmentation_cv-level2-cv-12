import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        use_batchnorm=True,
        attention_type=None
    ):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride, ceil_mode=True)
        self.conv = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention = md.Attention(attention_type, in_channels=in_channels)

    def forward(self, x):
        x = self.max_pool(x)    
        x = self.conv(x)
        x = self.attention(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        scale_factor, 
        use_batchnorm=True, 
        attention_type=None
        ) -> None:
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.conv = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.up_sample(x)    
        x = self.conv(x)
        x = self.attention(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x
    
class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class Unet3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        skip_channels,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        # computing blocks input and output channels
        head_channels = encoder_channels[-1]
        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[:-1]
        encoder_channels = encoder_channels[::-1]
        
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        
        # down 4
        self.x_1_down_4 = DownBlock(encoder_channels[3], skip_channels, 8, 8)
        self.x_2_down_4 = DownBlock(encoder_channels[2], skip_channels, 4, 4)
        self.x_3_down_4 = DownBlock(encoder_channels[1], skip_channels, 2, 2)
        
        # down 3
        self.x_1_down_3 = DownBlock(encoder_channels[3], skip_channels, 4, 4)
        self.x_2_down_3 = DownBlock(encoder_channels[2], skip_channels, 2, 2)
        
        # down 2
        self.x_1_down_2 = DownBlock(encoder_channels[3], skip_channels, 2, 2)
        
        # decoder 4
        self.x_5_up = UpBlock(head_channels, out_channels, 2)
        self.x_4 = ConvBlock(encoder_channels[0] + 3*skip_channels + out_channels, out_channels)
        # decoder 3
        self.x_5_2up = UpBlock(head_channels, out_channels, 4)
        self.x_4_up = UpBlock(out_channels, out_channels, 2)
        self.x_3 = ConvBlock(encoder_channels[1] + 2*skip_channels + out_channels*2, out_channels)
        # decoder 2
        self.x_5_3up = UpBlock(head_channels, out_channels, 8)
        self.x_4_2up = UpBlock(out_channels, out_channels, 4)
        self.x_3_up = UpBlock(out_channels, out_channels, 2)
        self.x_2 = ConvBlock(encoder_channels[2] + skip_channels + out_channels*3, out_channels)
        # decoder 1
        self.x_5_4up = UpBlock(head_channels, out_channels, 16)
        self.x_4_3up = UpBlock(out_channels, out_channels, 8)
        self.x_3_2up = UpBlock(out_channels, out_channels, 4)
        self.x_2_up = UpBlock(out_channels, out_channels, 2)
        self.x_1 = ConvBlock(encoder_channels[3] + out_channels*4, out_channels)

    def forward(self, *features):
        head = features[-1]
        skips = features[:-1]
        
        # center
        x = self.center(head)
        
        # down 4
        x_1_down = self.x_1_down_4(skips[0])
        x_2_down = self.x_2_down_4(skips[1])
        x_3_down = self.x_3_down_4(skips[2])
        
        # decoder 4
        x_5_up = self.x_5_up(x)
        x_4 = self.x_4(torch.cat([x_1_down, x_2_down, x_3_down, skips[3], x_5_up], dim=1))
        
        # down 3 
        x_1_down = self.x_1_down_3(skips[0])
        x_2_down = self.x_2_down_3(skips[1])
        
        # decoder 3
        x_5_2up = self.x_5_2up(x)
        x_4_up = self.x_4_up(x_4)
        x_3 = self.x_3(torch.cat([x_1_down, x_2_down, skips[2], x_4_up, x_5_2up], dim=1))
        
        # down 2
        x_1_down = self.x_1_down_2(skips[0])
        
        # decoder 2
        x_5_3up = self.x_5_3up(x)
        x_4_2up = self.x_4_2up(x_4)
        x_3_up = self.x_3_up(x_3)
        x_2 = self.x_2(torch.cat([x_1_down, skips[1], x_3_up, x_4_2up, x_5_3up], dim=1))
        
        # decoder 1
        x_5_4up = self.x_5_4up(x)
        x_4_3up = self.x_4_3up(x_4)
        x_3_2up = self.x_3_2up(x_3)
        x_2_up = self.x_2_up(x_2)
        x_1 = self.x_1(torch.cat([skips[0], x_2_up, x_3_2up, x_4_3up, x_5_4up], dim=1))
        
        return x_1