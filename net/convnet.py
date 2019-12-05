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

def dense(in_dim, out_dim, hidden_dim, n_dense):
    layers = [None] * (2*n_dense + 1)
    shapes = [in_dim] + [hidden_dim for i in range(n_dense)] + [out_dim]
    layers[::2] = [nn.Linear(shapes[i],shapes[i+1]) for i in range(len(shapes)-1)]
    layers[1::2] = [nn.ReLU(True) for i in range(n_dense)]
    return nn.Sequential(*layers)

class ConvNet(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.in_channels = args.in_channels
        self.out_features = args.num_way
        self.hidden_channels = args.hidden_channels

        self.encoder = [conv3x3(self.in_channels, self.hidden_channels, True)] +  [conv3x3(self.hidden_channels, self.hidden_channels, i<3) for i in range(args.n_conv-1)]
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = dense(self.hidden_channels*5*5, self.out_features, args.hidden_dim, args.n_dense)
        self.init_params()
        return None
    
    def init_params(self):
        for k, v in self.named_parameters():
            if ('Conv' in k) or ('Linear' in k):
                if ('weight' in k):
                    nn.init.kaiming_uniform_(v)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
            elif ('Batch' in k):
                if ('weight' in k):
                    nn.init.constant_(v, 1.0)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
        return None

    def forward(self, x):

        x = self.encoder(x) # (N, 3, 80, 80) -> (N, 64, 5, 5)

        x = self.decoder(x.reshape(x.shape[0], -1)) # (N, out_features)

        return x