import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, use_maxpool=False):
    block = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    ]
    if use_maxpool:
        block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)

class ConvNet(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.in_channels = args.in_channels
        self.out_features = args.num_way
        self.hidden_channels = args.hidden_channels
        
        self.encoder = nn.Sequential(
            conv3x3(self.in_channels, self.hidden_channels, True),
            conv3x3(self.hidden_channels, self.hidden_channels, True),
            conv3x3(self.hidden_channels, self.hidden_channels, True),
            conv3x3(self.hidden_channels, self.hidden_channels, True),
            # conv3x3(hidden_channels, hidden_channels),
            # conv3x3(hidden_channels, hidden_channels),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_channels * 5 * 5, self.out_features),
        )
        self.init_params()
        return None
    
    def init_params(self):
        for k, v in self.named_parameters():
            if ('conv' in k) or ('fc' in k):
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

    def forward(self, x):

        x = self.encoder(x) # (N, 3, 80, 80) -> (N, 64, 5, 5)

        x = self.decoder(x.reshape(x.shape[0], -1)) # (N, out_features)

        return x