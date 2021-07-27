import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from world_model.layers import RestrictionActivation, ActivatedNormLinear, Upsampling, ConvBlock, Bottleneck, \
    UpsamplingConcat
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
    def __init__(self, in_channels=64, out_channels=256, downsample_factor=8, name='efficientnet-b0'):
        super().__init__()
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor
        self.version = name.split('-')[1]

        if self.downsample_factor == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample_factor == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample_factor} not handled.')

        self.backbone = EfficientNet.from_pretrained(name)
        self.backbone._conv_stem = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, bias=False, padding=1
        )
        self.delete_unused_layers()

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(upsampling_out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample_factor == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

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

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample_factor == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample_factor == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']

        x = self.upsampling_layer(input_1, input_2)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector

        return self.last_layer(x)


class RepresentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ActivatedNormLinear(in_channels, 256),
            ActivatedNormLinear(256, 128),
            ActivatedNormLinear(128, 128),
            ActivatedNormLinear(128, out_channels),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        return self.model(x)


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
    def __init__(self, in_channels, out_channels=9):
        super().__init__()
        self.model = nn.Sequential(
            Upsampling(in_channels, 512),
            Bottleneck(512, 512),
            Upsampling(512, 256),
            Bottleneck(256, 256),
            Upsampling(256, 128),
            Bottleneck(128, 128),
            Upsampling(128, 64),
            Bottleneck(64, 64),
            Upsampling(64, 32),
            Bottleneck(32, 32),
            nn.Conv2d(32, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # Spatially-broadcast vector to (C, 8, 8) tensor
        b, s, c = x.shape
        x = x.view(b*s, c, 1, 1)
        x = x.expand(-1, -1, 4, 4)

        x = self.model(x)
        return x.view(b, s, *x.shape[1:])


class RSSM(nn.Module):
    def __init__(self, encoder_dim, action_dim, state_dim, hidden_state_dim):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_state_dim = hidden_state_dim

        self.pre_gru_net = RepresentationModel(in_channels=state_dim + action_dim, out_channels=state_dim + action_dim)

        self.recurrent_model = nn.GRUCell(
            input_size=state_dim + action_dim,
            hidden_size=hidden_state_dim,
        )

        self.posterior = RepresentationModel(
            in_channels=hidden_state_dim + encoder_dim,
            out_channels=2*state_dim,
        )

        self.prior = RepresentationModel(in_channels=hidden_state_dim, out_channels=2*state_dim)

    def forward(self, input_embedding, action, deployment=False):
        """
        Inputs
        ------
            input_embedding: torch.Tensor size (B, S, C)
            action: torch.Tensor size (B, S, 2)

        Returns
        -------
            dict:
                hidden_states: torch.Tensor (B, S, C_h)
                samples: torch.Tensor (B, S, C_s)
                posterior_mu: torch.Tensor (B, S, C_s)
                posterior_sigma: torch.Tensor (B, S, C_s)
                prior_mu: torch.Tensor (B, S, C_s)
                prior_sigma: torch.Tensor (B, S, C_s)
        """
        batch_size, sequence_length = input_embedding.shape[:2]

        h = []
        sample = []
        z_mu = []
        z_sigma = []
        z_hat_mu = []
        z_hat_sigma = []

        # Initialisation
        h_t = input_embedding.new_zeros((batch_size, self.hidden_state_dim))
        sample_t = input_embedding.new_zeros((batch_size, self.state_dim))
        for t in range(sequence_length):
            if t == 0:
                action_t = torch.zeros_like(action[:, 0])
            else:
                action_t = action[:, t-1]
            output = self.observe_step(
                h_t, sample_t, action_t, input_embedding[:, t]
            )

            h.append(output['prior']['h_t'])
            sample.append(output['posterior']['sample_t'])
            z_mu.append(output['posterior']['mu'])
            z_sigma.append(output['posterior']['log_sigma'])
            z_hat_mu.append(output['prior']['mu'])
            z_hat_sigma.append(output['prior']['log_sigma'])

        h = torch.stack(h, dim=1)
        sample = torch.stack(sample, dim=1)
        z_mu = torch.stack(z_mu, dim=1)
        z_sigma = torch.stack(z_sigma, dim=1)
        z_hat_mu = torch.stack(z_hat_mu, dim=1)
        z_hat_sigma = torch.stack(z_hat_sigma, dim=1)

        return h, sample, z_mu, z_sigma, z_hat_mu, z_hat_sigma

    def imagine_step(self, h_t, sample_t, action_t):
        input_t = torch.cat([sample_t, action_t], dim=-1)
        input_t = self.pre_gru_net(input_t)
        h_t = self.recurrent_model(input_t, h_t)

        z_t_hat = self.prior(h_t)
        z_t_hat_mu, z_t_hat_sigma = torch.split(z_t_hat, z_t_hat.shape[-1] // 2, dim=-1)
        sample_t = self.sample_from_distribution(z_t_hat_mu, z_t_hat_sigma)
        imagine_output = {
            'h_t': h_t,
            'sample_t': sample_t,
            'mu': z_t_hat_mu,
            'log_sigma': z_t_hat_sigma,
        }
        return imagine_output

    def observe_step(self, h_t, sample_t, action_t, embedding_t):
        imagine_output = self.imagine_step(h_t, sample_t, action_t)

        z_t = self.posterior(torch.cat([imagine_output['h_t'], embedding_t], dim=-1))

        z_t_mu, z_t_sigma = torch.split(z_t, z_t.shape[-1] // 2, dim=-1)

        sample_t = self.sample_from_distribution(z_t_mu, z_t_sigma)

        posterior_output = {
            'sample_t': sample_t,
            'mu': z_t_mu,
            'log_sigma': z_t_sigma,
        }

        output = {
            'prior': imagine_output,
            'posterior': posterior_output,
        }

        return output

    def sample_from_distribution(self, mu, log_sigma):
        noise = torch.randn_like(mu)
        sample = mu + torch.exp(log_sigma) * noise
        return sample


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
