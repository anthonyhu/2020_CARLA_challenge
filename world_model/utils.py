import numpy as np
import torch
import torch.nn as nn

from carla_project.src.common import CONVERTER
from carla_project.src.common import COLOR

N_CLASSES = len(COLOR)

# https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/local_planner.py
ROUTE_COMMANDS = {1: 'LEFT',
                  2: 'RIGHT',
                  3: 'STRAIGHT',
                  4: 'LANEFOLLOW',
                  5: 'CHANGELANELEFT',
                  6: 'CHANGELANERIGHT',
                  }


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def preprocess_semantic(semantic_np):
    topdown = CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


def preprocess_bev_state(x):
    # Crop bottom
    x = x.crop((128, 0, 128 + 256, 256))
    x = np.array(x)
    x = preprocess_semantic(x)
    # Crop top and sides
    x_out = x[..., 128:, 64:-64]

    # Double size
    x_out = nn.functional.interpolate(x_out[None], scale_factor=2, mode='nearest')[0]
    return x_out
