import torch.nn as nn

class DeconvNet_JJ(nn.Module):
    def __init__(self, num_classes=11, num_convs=[2, 2, 3, 3, 3]):
        super(DeconvNet_JJ, self).__init__()
        
        self.encoders = nn.ModuleList()
        max_pool = nn.MaxPool2d(2, 2, ceil_mode=True, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(2, 2)
        in_channels = 3
        out_channels = 64
        
        for num_conv in num_convs:
            conv_block = []
            for _ in range(num_conv):
                conv_block.append(
                    self.CBR(in_channels, out_channels, 3, 1, 1)
                )
                in_channels = out_channels
            
            conv_block.append(max_pool)
            self.encoders.append(nn.Sequential(*conv_block))
            
            if out_channels < 512:
                out_channels *= 2
        
        out_channels = 4096
        drop_out = nn.Dropout2d()
        self.middle_block = nn.Sequential(
            self.CBR(in_channels, out_channels, 7, 1, 0),
            drop_out,
            self.CBR(out_channels, out_channels, 1, 1, 0),
            drop_out,
            self.DCB(out_channels, in_channels, 7, 1, 0)
        )
        
        self.decoders = nn.ModuleList()
        out_channels = 512
        
        for idx, num_conv in enumerate(num_convs[::-1]):
            deconv_block = []
            for _ in range(num_conv):
                deconv_block.append(
                    self.DCB(in_channels, out_channels, 3, 1, 1)
                )
                in_channels = out_channels
            
            self.decoders.append(nn.Sequential(*deconv_block))
            
            if out_channels > 64:
                out_channels //= 2
                
        self.classifier = nn.Conv2d(out_channels, num_classes, 1, 1, 0)
    
    def forward(self, x):
        pool_indices = []
        for encoder in self.encoders:
            x, i = encoder(x)
            pool_indices.append(i)
        
        x = self.middle_block(x)
        
        for decoder, i in zip(self.decoders, pool_indices[::-1]):
            x = self.max_unpool(x, i)
            x = decoder(x)
        
        x = self.classifier(x)
                
        return x
    
    def CBR(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def DCB(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )