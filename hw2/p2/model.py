import torch


class ResBlock(torch.nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample=None):

        super().__init__()

        self.stride = stride
        self.downsample = downsample

        self.conv_x1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_x1 = torch.nn.BatchNorm2d(out_channels)
        
        self.conv_x2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_x2 = torch.nn.BatchNorm2d(out_channels)

        self.activation = torch.nn.GELU()

    def forward(self, x):
        
        out = self.conv_x1(x)
        out = self.bn_x1(out)
        out = self.activation(out)

        out = self.conv_x2(out)
        out = self.bn_x2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.activation(out)

        return out


class ResNet(torch.nn.Module):

    expansion = 1

    def __init__(self, block, block_layers, img_channels, num_classes):

        super().__init__()

        self.in_channels = 64

        self.conv1 = torch.nn.Conv2d(img_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.in_channels)
        self.activation = torch.nn.GELU()
        self.first_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.conv2_x = self._res_layer(block, block_layers[0], 64, stride=1)
        self.conv3_x = self._res_layer(block, block_layers[1], 128, stride=2)
        self.conv4_x = self._res_layer(block, block_layers[2], 256, stride=2)
        self.conv5_x = self._res_layer(block, block_layers[3], 512, stride=2)

        self.last_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(512, num_classes)
    
    def forward(self, x, return_feats=False):
        
        res_out = self.conv1(x)
        res_out = self.bn1(res_out)
        res_out = self.activation(res_out)
        res_out = self.first_pool(res_out)

        res_out = self.conv2_x(res_out)
        res_out = self.conv3_x(res_out)
        res_out = self.conv4_x(res_out)
        res_out = self.conv5_x(res_out)

        res_out = self.last_pool(res_out)
        res_out = self.flatten(res_out)
        
        # image resnet extracted features for verification
        if return_feats:
            return res_out
        
        # classification
        out = self.linear(res_out)

        return out
    
    def _res_layer(self, block, num_blocks, out_channels, stride):

        downsample = None

        if stride != 1 or out_channels != self.in_channels:
            downsample = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                             torch.nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))

        self.in_channels = out_channels

        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return torch.nn.Sequential(*layers)


class ConvNextStem(torch.nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.stem = torch.nn.Sequential(torch.nn.Conv2d(in_channels, 96, kernel_size=4, stride=4),
                                        torch.nn.BatchNorm2d(96))
    
    def forward(self, x):

        return self.stem(x)


class ConvNextBlock(torch.nn.Module):

    factor = 4

    def __init__(self, in_channels, out_channels, stride):
    
        super().__init__()

        in_1  = in_channels
        out_1 = out_channels

        in_2  = out_channels
        out_2 = out_channels * self.factor

        in_3  = out_channels * self.factor
        out_3 = out_channels

        self.resx = torch.nn.Sequential(torch.nn.Conv2d(in_1, out_1, kernel_size=7, stride=stride, padding=3, groups=in_1),
                                        torch.nn.BatchNorm2d(out_1),
                                        
                                        torch.nn.Conv2d(in_2, out_2, kernel_size=1, stride=1),
                                        torch.nn.GELU(),
                                        
                                        torch.nn.Conv2d(in_3, out_3, kernel_size=1, stride=1))
        
        self.shortcut = torch.nn.Identity()
        
        if in_channels != out_channels or stride != 1:

            self.shortcut = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                                                torch.nn.BatchNorm2d(out_channels))
    
    def forward(self, x):

        out = self.resx(x)
        
        x = self.shortcut(x)

        return out + x


class ConvNext(torch.nn.Module):

    def __init__(self, in_channels, num_classes, layer_count):

        super().__init__()

        self.backbone = torch.nn.Sequential(ConvNextStem(in_channels=in_channels),
                                            self._stage(layer_count[0], 96 , 96 , stride=1),
                                            self._stage(layer_count[1], 96 , 192, stride=2),
                                            self._stage(layer_count[2], 192, 384, stride=2),
                                            self._stage(layer_count[3], 384, 768, stride=2))
        
        self.last_pool     = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        self.flatten  = torch.nn.Flatten()
        self.linear   = torch.nn.Linear(768, num_classes)
    
    def _stage(self, count, in_channels, out_channels, stride):

        return torch.nn.Sequential(ConvNextBlock(in_channels, out_channels, stride=stride),
                                   *[ConvNextBlock(out_channels, out_channels, stride=1) for _ in range(count-1)])
    
    def forward(self, x, return_feats=False):

        cnn_out = self.backbone(x)

        if return_feats:

            return cnn_out
        
        out = self.last_pool(cnn_out)
        out = self.flatten(out)
        out = self.linear(out)

        return out


def get_ResNet34(in_channels, num_classes):

    return ResNet(ResBlock, [3, 4, 6, 3], img_channels=in_channels, num_classes=num_classes)


def get_ConvNext(in_channels, num_classes):

    return ConvNext(in_channels, num_classes, layer_count=[3, 3, 9, 3])
