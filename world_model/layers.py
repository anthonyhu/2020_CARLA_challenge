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


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, action_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels + action_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_skip, action):
        # Spatially broadcast
        b, _, h, w = x.shape
        action = action.view(b, -1, 1, 1).expand(b, -1, h, w)
        x = torch.cat([x, action], dim=1)
        x = self.upsample_layer(x)
        return x + x_skip


class ActivatedNormLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = nn.Sequential(nn.Linear(in_channels, out_channels),
                                    nn.BatchNorm1d(out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        return self.module(x)
