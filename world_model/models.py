import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from world_model.layers import RestrictionActivation


class Policy(nn.Module):
    def __init__(self, in_channels=64, out_channels=4, command_channels=6, speed_as_input=False,
                 name='efficientnet-b0'):
        super().__init__()
        self.command_channels = command_channels
        self.speed_as_input = speed_as_input

        self.backbone = EfficientNet.from_pretrained(name)
        self.backbone._conv_stem = nn.Conv2d(
            in_channels + command_channels, 32, kernel_size=3, stride=2, bias=False, padding=1
        )

        self.output_net = nn.Sequential(nn.Conv2d(320, 512, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(512, momentum=0.01),
                                        self.backbone._swish,
                                        nn.AdaptiveAvgPool2d(output_size=1),
                                        nn.Flatten(),
                                        ActivatedNormLinear(512, 256),
                                        ActivatedNormLinear(256, 128),
                                        ActivatedNormLinear(128, 64),
                                        )
        speed_channels = 1 if self.speed_as_input else 0
        self.last_layer = nn.Linear(64 + command_channels + speed_channels, out_channels)

        self.steering_activation = RestrictionActivation(min_value=-1, max_value=1)
        self.throttle_activation = RestrictionActivation(min_value=0, max_value=0.75)

        self.delete_unused_layers()

    def delete_unused_layers(self):
        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        return x

    def forward(self, x, route_commands, speed):
        # concatenate route_commands
        b, c, h, w = x.shape

        # substract 1 because commands start at 1.
        route_commands = torch.nn.functional.one_hot(route_commands.squeeze(-1) - 1, self.command_channels)

        route_commands_1 = route_commands.view(b, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([x, route_commands_1], dim=1)

        x = self.get_features(x)  # get feature vector
        x = self.output_net(x)
        x_concat = torch.cat([x, route_commands], dim=-1)
        if self.speed_as_input:
            x_concat = torch.cat([x_concat, speed], dim=-1)
        x = self.last_layer(x_concat)

        # Restrict steering and throttle output range
        x[..., 0] = self.steering_activation(x[..., 0])
        x[..., 1] = self.throttle_activation(x[..., 1])

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


class TransitionModel(nn.Module):
    def __init__(self, in_channels, action_channels=4):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, action_channels, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, action_channels, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, action_channels, shared_out_channels, scale_factor=2)

        self.output_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, in_channels, kernel_size=1, padding=0),
        )

    def forward(self, x, action):
        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/2, W/2)
        x = self.layer1(x)
        skip_x['2'] = x
        # (H/4, W/4)
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'], action)

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'], action)

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'], action)

        output = self.output_head(x)
        return output


class ActivatedNormLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = nn.Sequential(nn.Linear(in_channels, out_channels),
                                    nn.BatchNorm1d(out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        return self.module(x)


class RewardModel(Policy):
    def __init__(self, in_channels=64, encoding_channels=16, name='efficientnet-b0', trajectory_length=10, n_actions=4):
        super().__init__(in_channels=in_channels, out_channels=encoding_channels, name=name)

        self.output_net = nn.Sequential(nn.Conv2d(320, 512, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(512, momentum=0.01),
                                        self.backbone._swish,
                                        nn.AdaptiveAvgPool2d(output_size=1),
                                        nn.Flatten(),
                                        nn.Linear(512, encoding_channels),
                                        )

        self.reward_net = nn.Sequential(ActivatedNormLinear(encoding_channels + trajectory_length*n_actions, 512),
                                        ActivatedNormLinear(512, 256),
                                        ActivatedNormLinear(256, 128),
                                        ActivatedNormLinear(128, 64),
                                        nn.Linear(64, 1),
                                        )

    def forward(self, state, trajectory):
        """
        Inputs
        ------
            state: torch.Tensor (B, C, H, W)
            trajectory: torch.Tensor (B, T, 4)
                predicted future trajectory of the ego car
        """
        state = self.get_features(state)  # get feature vector
        state_encoding = self.output_net(state)

        batch_size = trajectory.shape[0]
        trajectory = trajectory.view(batch_size, -1)

        joint_feature = torch.cat([state_encoding, trajectory], dim=-1)
        reward = self.reward_net(joint_feature)
        return reward
