# ResNet Wide Version as in Qiao's Paper
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        return None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.in_channels = args.in_channels
        self.out_features = args.num_way

        cfg = [160, 320, 640]
        layers = [3,3,3]
        self.inplanes = iChannels = int(cfg[0]/2)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(iChannels),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, cfg[0], layers[0], stride=2),
            self._make_layer(BasicBlock, cfg[1], layers[1], stride=2),
            self._make_layer(BasicBlock, cfg[2], layers[2], stride=2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg[2] * 10 * 10, self.out_features),
        )
        self.init_params()
        return None

    def init_params(self):
        for k, v in self.named_parameters():
            if ('conv' in k) or ('meta' in k):
                if ('weight' in k):
                    nn.init.kaiming_uniform_(v)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
            elif ('bn' in k):
                if ('weight' in k):
                    nn.init.constant_(v, 1.0)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
        return None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, is_pre=True):
        
        x = self.encoder(x) # (N, 3, 80, 80) -> (N, 640, 10, 10)

        x = self.decoder(x.reshape(x.shape[0], -1)) # (N, out_features)
        
        return x
