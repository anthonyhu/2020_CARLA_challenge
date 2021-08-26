import numpy as np
import torch
import torch.nn as nn
import torchvision

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
    #x_out = nn.functional.interpolate(x_out[None], scale_factor=2, mode='nearest')[0]
    return x_out


def preprocess_batch(batch, device, unsqueeze=False):
    for key, value in batch.items():
        batch[key] = value.to(device)
        if unsqueeze:
            batch[key] = batch[key].unsqueeze(0)


def pack_sequence_dim(x):
    b, s = x.shape[:2]
    return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s):
    return x.view(b, s, *x.shape[1:])


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height
    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
