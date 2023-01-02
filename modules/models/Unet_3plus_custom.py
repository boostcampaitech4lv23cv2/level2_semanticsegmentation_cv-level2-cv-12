import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

class UNet_3plus_custom(nn.Module):
    def __init__(self, depth=5) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.max_pool = nn.MaxPool2d
        self.max_unpool = nn.MaxUnpool2d
        self.interpolation = F.interpolate
        
        in_channels, out_channels = 3, 64
        connection_channels = []
        
        for i in range(depth):
            if i == 0:
                self.encoders.append(self.double_CBR(in_channels, out_channels, 3, 1, 1))
            else:
                self.encoders.append(self.double_CBR(in_channels, out_channels, 3, 1, 1).append(self.max_pool(2, 2, ceil_mode=True, return_indices=True)))
            in_channels = out_channels
            connection_channels.append(out_channels)
            out_channels *= 2
        
        last_encoder_channels = out_channels // 2
                
        connection_channels = connection_channels[:-1][::-1]

        out_channels = 320
        for i in range(depth - 1):
            self.decoders.append(self.double_DBR(sum(connection_channels[i:]) + out_channels*i + last_encoder_channels, out_channels, 3, 1, 1))
            
        self.cgm = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(1024, 11, 1, 1, 0),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Conv2d(320, 11, 3, 1, 1)
        
        self.init_params()
            
    def forward(self, x):
        inters, intras = [], []
        pool_indices = []
            
        for i, encoder in enumerate(self.encoders[:-1]):
            if i == 0:
                x = encoder(x)
            else:
                x, indices = encoder(x)
                pool_indices.append(indices)
            inters.append(x)
        
        last_encoder_x, las_encoder_indices = self.encoders[-1](x)
        
        inters = inters[::-1]
        
        for i, decoder in enumerate(self.decoders):
            connections = [inters[i]]
            for j, connection in enumerate(inters[i+1:]):
                connection = self.max_pool(2**(j+1), 2**(j+1))(connection)
                connections.append(connection)
            
            last_encoder_x_temp = self.max_unpool(2**(i+1), 2**(i+1)).to('cuda')(last_encoder_x, las_encoder_indices)
            last_encoder_x_temp = self.double_DBR(1024, 1024, 3, 1, 1).to('cuda')(last_encoder_x_temp)
            connections.append(last_encoder_x_temp)
            
            if intras:
                for j, intra in enumerate(intras[::-1]):
                    _, _, h, w = intra.size()
                    intra = self.interpolation(intra, (h*2**(j+1), w*2**(j+1)), mode="bilinear")
                    connections.append(intra)
                    
            x = torch.cat(connections, dim=1)
            x = decoder(x)

            intras.append(x)
        
        x = self.classifier(x)
        
        return x 
    
    def double_CBR(self, in_channels, out_channels, kernel_size, stride, padding):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )
        
        return conv_block
    
    def double_DBR(self, in_channels, out_channels, kernel_size, stride, padding):
        trans_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )
        
        return trans_conv_block
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from torchsummary import summary as summary
    from Efficientb4_unet_pp import Efficientb4_UNet_PP
    from torchvision.models import vgg11
    
    # model = Efficientb4_UNet_PP().to(device)
    model = vgg11().cuda()
    
    x = torch.rand((3, 512, 512))
    
    summary(model, (3, 512, 512))

# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from torch.nn import init

# class Unet_3plus(nn.Module):
#     def __init__(self, depth=5) -> None:
#         super().__init__()
#         self.encoders = nn.ModuleList()
#         self.decoders = nn.ModuleList()
#         self.max_pool = nn.MaxPool2d
#         self.max_unpool = nn.MaxUnpool2d
#         self.interpolation = F.interpolate
        
#         in_channels, out_channels = 3, 64
#         connection_channels = []
        
#         for i in range(depth):
#             if i == 0:
#                 self.encoders.append(self.double_CBR(in_channels, out_channels, 3, 1, 1))
#             else:
#                 self.encoders.append(self.double_CBR(in_channels, out_channels, 3, 1, 1).append(self.max_pool(2, 2, ceil_mode=True)))
#             in_channels = out_channels
#             connection_channels.append(out_channels)
#             out_channels *= 2
        
#         last_encoder_channels = out_channels // 2
                
#         connection_channels = connection_channels[:-1][::-1]

#         out_channels = 320
#         for i in range(depth - 1):
#             self.decoders.append(self.double_DBR(sum(connection_channels[i:]) + out_channels*i + last_encoder_channels, out_channels, 3, 1, 1))
            
#         self.cgm = nn.Sequential(
#             nn.Dropout2d(),
#             nn.Conv2d(1024, 11, 1, 1, 0),
#             nn.AdaptiveMaxPool2d(1),
#             nn.Sigmoid()
#         )
        
#         self.classifier = nn.Conv2d(320, 11, 3, 1, 1)
        
#         self.init_params()
            
#     def forward(self, x):
#         inters, intras = [], []
            
#         for i, encoder in enumerate(self.encoders[:-1]):
#             x = encoder(x)
#             inters.append(x)
        
#         last_encoder_x = self.encoders[-1](x)
#         _, _, height, width = last_encoder_x.size()
        
#         inters = inters[::-1]
        
#         for i, decoder in enumerate(self.decoders):
#             connections = [inters[i]]
#             for j, connection in enumerate(inters[i+1:]):
#                 connection = self.max_pool(2**(j+1), 2**(j+1))(connection)
#                 connections.append(connection)
            
#             last_encoder_x_temp  = self.interpolation(last_encoder_x, (height*2**(i+1), width*2**(i+1)), mode="bilinear")
#             last_encoder_x_temp = self.double_DBR(1024, 1024, 3, 1, 1).to('cuda')(last_encoder_x_temp)
#             connections.append(last_encoder_x_temp)
            
#             if intras:
#                 for j, intra in enumerate(intras[::-1]):
#                     _, _, h, w = intra.size()
#                     intra = self.interpolation(intra, (h*2**(j+1), w*2**(j+1)), mode="bilinear")
#                     connections.append(intra)
                    
#             x = torch.cat(connections, dim=1)
#             x = decoder(x)

#             intras.append(x)
        
#         x = self.classifier(x)
        
#         return x 
    
#     def double_CBR(self, in_channels, out_channels, kernel_size, stride, padding):
#         conv_block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             # nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
#             # nn.BatchNorm2d(out_channels),
#             # nn.ReLU()
#         )
        
#         return conv_block
    
#     def double_DBR(self, in_channels, out_channels, kernel_size, stride, padding):
#         trans_conv_block = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             # nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),
#             # nn.BatchNorm2d(out_channels),
#             # nn.ReLU()
#         )
        
#         return trans_conv_block
    
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.normal_(m.weight.data, 1.0, 0.02)
#                 init.constant_(m.bias.data, 0.0)
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     from torchsummary import summary as summary
#     from Efficientb4_unet_pp import Efficientb4_UNet_PP
#     from torchvision.models import vgg11
    
#     # model = Efficientb4_UNet_PP().to(device)
#     model = Unet_3plus().to('cuda')
#     x = torch.rand((3, 512, 512))
    
#     summary(model, (3, 512, 512))