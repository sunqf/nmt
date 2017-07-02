
import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.Tensor(1))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.epsilon = epsilon

    def reset_parameters(self):
        self.scale.data.fill_(1.0)
        self.bias.data.fill_(0.0)

    def forward(self, input):
        mean = input.mean(-1)
        variance = input.var(-1) + self.epsilon
        norm = (input - mean) / torch.sqrt(variance)
        return norm * self.scale + self.bias
