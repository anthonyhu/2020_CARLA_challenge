import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from world_model.layers import RestrictionActivation, ActivatedNormLinear, UpsamplingAdd
from world_model.temporal_layers import Bottleneck3D, TemporalBlock


class TemporalModel(nn.Module):
    def __init__(
            self, in_channels, receptive_field, input_shape, start_out_channels=64, extra_in_channels=0,
            n_spatial_layers_between_temporal_layers=0, use_pyramid_pooling=True):
        super().__init__()
        self.receptive_field = receptive_field
        n_temporal_layers = receptive_field - 1

        h, w = input_shape
        modules = []

        block_in_channels = in_channels
        block_out_channels = start_out_channels

        for _ in range(n_temporal_layers):
            if use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, h, w)]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                block_in_channels,
                block_out_channels,
                use_pyramid_pooling=use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(block_out_channels, block_out_channels, kernel_size=(1, 3, 3))
                for _ in range(n_spatial_layers_between_temporal_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += extra_in_channels

        self.out_channels = block_in_channels

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # Reshape input tensor to (batch, C, time, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x[:, (self.receptive_field - 1):].contiguous()


class TemporalModelIdentity(nn.Module):
    def __init__(self, in_channels, receptive_field):
        super().__init__()
        self.receptive_field = receptive_field
        self.out_channels = in_channels

    def forward(self, x):
        return x[:, (self.receptive_field - 1):].contiguous()


class Encoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=256, name='efficientnet-b0'):
        super().__init__()

        self.backbone = EfficientNet.from_pretrained(name)
        self.backbone._conv_stem = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, bias=False, padding=1
        )

        self.output_net = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(320, out_channels, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels, momentum=0.01),
                                        self.backbone._swish,
                                        )

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

    def forward(self, x):
        x = self.get_features(x)
        x = self.output_net(x)
        x = x.mean(dim=(-1, -2))
        return x


class RepresentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ActivatedNormLinear(in_channels, out_channels),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        b, s, c = x.shape
        x = x.view(b*s, c)
        x = self.model(x)
        return x.view(b, s, x.shape[-1])


class Policy(nn.Module):
    def __init__(self, in_channels=64, out_channels=4, command_channels=6, speed_as_input=False,
                 name='efficientnet-b0'):
        super().__init__()
        self.command_channels = command_channels
        self.speed_as_input = speed_as_input

        speed_channels = 1 if self.speed_as_input else 0

        self.module = nn.Sequential(
            nn.Conv2d(in_channels + command_channels + speed_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            ActivatedNormLinear(512, 128),
            nn.Linear(128, out_channels)
        )


        #self.steering_activation = RestrictionActivation(min_value=-1, max_value=1)
        #self.throttle_activation = RestrictionActivation(min_value=0, max_value=0.75)

    def forward(self, x, route_commands, speed):
        # concatenate route_commands
        b, c, h, w = x.shape

        # substract 1 because route commands start at 1.
        route_commands = torch.nn.functional.one_hot(route_commands.squeeze(-1) - 1, self.command_channels)

        route_commands = route_commands.view(b, -1, 1, 1).expand(-1, -1, h, w)
        speed = speed.view(b, -1, 1, 1).expand(-1, -1, h, w)

        x_concat = torch.cat([x, route_commands], dim=1)
        if self.speed_as_input:
            x_concat = torch.cat([x_concat, speed], dim=1)
        x = self.module(x_concat)

        # Restrict steering and throttle output range
        #x[..., 0] = self.steering_activation(x[..., 0])
        #x[..., 1] = self.throttle_activation(x[..., 1])

        return x


class Distribution(nn.Module):
    def __init__(self, in_channels=512, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.module = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            ActivatedNormLinear(512, 256),
            ActivatedNormLinear(256, 128),
            nn.Linear(128, 2*latent_dim)
        )

    def forward(self, x):
        output = self.module(x)
        mean, std = output[..., :self.latent_dim].contiguous(), output[..., self.latent_dim:].contiguous()
        return mean, std


class Decoder(nn.Module):
    def __init__(self, in_channels, action_channels=4, out_channels=256):
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
            nn.Conv2d(shared_out_channels, out_channels, kernel_size=1, padding=0),
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
