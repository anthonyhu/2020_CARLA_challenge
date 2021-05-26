import torch
import torch.nn as nn


class RestrictionActivation(nn.Module):
    """ Constrain output to be between min_value and max_value."""

    def __init__(self, min_value=0, max_value=1):
        super().__init__()
        self.scale = (max_value - min_value) / 2
        self.offset = min_value

    def forward(self, x):
        x = torch.tanh(x) + 1  # in range [0, 2]
        x = self.scale * x + self.offset  # in range [min_value, max_value]
        return x
