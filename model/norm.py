
import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon

    def forward(self, input):
        mean = input.mean(-1)
        std = input.std(-1)
        norm = (input - mean.expand_as(input)) / (std.expand_as(input) + self.epsilon)
        return self.alpha.expand_as(input) * norm + self.beta.expand_as(input)