import torch.nn as nn

class FCN16s(nn.Module):
    def __init__(self, num_classes=11, num_classifier=3, num_blocks=[2, 2, 3, 3, 3]):
        super(FCN16s, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.blocks = nn.ModuleList()
        in_channels = 3
        out_channels = 64
        
        for idx, num_block in enumerate(num_blocks):
            conv_block = []
            for i in range(num_block):
                conv_block.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
                conv_block.append(nn.BatchNorm2d(out_channels))
                conv_block.append(self.relu)
                in_channels = out_channels
            
            if out_channels < 512:
                out_channels *= 2
            
            conv_block.append(self.max_pool)
            self.blocks.append(nn.Sequential(*conv_block))
        
        self.temp_conv = nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        
        out_channels = 4096
        classifiers = []
        for i in range(num_classifier):
            if i == num_classifier - 1:
                classifiers.append(nn.Conv2d(in_channels, num_classes, 1, 1, 0))
            else:
                classifiers.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
            in_channels=out_channels

        self.classifier = nn.Sequential(*classifiers)
        self.classifier_trans_conv = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1)
        self.final_trans_conv = nn.ConvTranspose2d(num_classes, num_classes, 32, 16, 8)
        
    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
            
            if idx == 3:
                temp = self.temp_conv(x)
        
        x = self.classifier(x)
        x = self.classifier_trans_conv(x)
        x = self.final_trans_conv(x + temp)
        
        return x