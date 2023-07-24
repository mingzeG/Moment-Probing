import torch
from torch import nn
from torch.nn.parameter import Parameter

class PSRP(nn.Module):
    def __init__(self, channel=768, reduction=16):
        super(PSRP, self).__init__()
        
        self.down = nn.Linear(channel, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up_weight = nn.Linear(reduction, channel, bias=False)
        self.up_bias = nn.Linear(reduction, channel, bias=False)

    def forward(self, x):
        weight = self.up_weight(self.relu(self.down(x)))
        bias = self.up_bias(self.relu(self.down(x)))

        return weight, bias