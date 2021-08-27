import numpy as np
import torch
import torch.nn as nn

from world_model.layers.layers import VoxelsSumming
from world_model.utils import calculate_birds_eye_view_parameters, pack_sequence_dim, unpack_sequence_dim
from world_model.models.models import Encoder, DownsampleFeatures, RSSM, RewardModel, Decoder


class Fiery(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.intrinsics, self.extrinsics = self.calculate_geometry()

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.config.LIFT.X_BOUND, self.config.LIFT.Y_BOUND, self.config.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.config.MODEL.ENCODER.DOWNSAMPLE_FACTOR
        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.config.LIFT.X_BOUND[1], self.config.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(config=self.config.MODEL.ENCODER, D=self.depth_channels)
        self.downsample_module = DownsampleFeatures(self.config.MODEL.ENCODER.OUTPUT_DIM)

        # Recurrent model
        self.receptive_field = self.config.RECEPTIVE_FIELD

        # Recurrent state sequence module
        self.rssm = RSSM(
            encoder_dim=self.config.MODEL.ENCODER.OUTPUT_DIM,
            action_dim=self.config.MODEL.ACTION_DIM,
            state_dim=self.config.MODEL.RECURRENT_MODEL.STATE_DIM,
            hidden_state_dim=self.config.MODEL.RECURRENT_MODEL.HIDDEN_STATE_DIM,
            receptive_field=self.receptive_field,
        )

        if self.config.MODEL.TRANSITION.ENABLED:
            self.decoder = Decoder(
                in_channels=self.config.MODEL.RECURRENT_MODEL.HIDDEN_STATE_DIM + self.config.MODEL.RECURRENT_MODEL.STATE_DIM,
                out_channels=self.config.SEMANTIC_SEG.N_CHANNELS,
            )

        if self.config.MODEL.REWARD.ENABLED:
            print('Enabled: Reward')
            self.reward_model = RewardModel(
                in_channels=self.config.MODEL.IN_CHANNELS,
                trajectory_length=self.config.SEQUENCE_LENGTH,
                n_actions=self.config.MODEL.ACTION_DIM,
            )

        # self.policy = Policy(in_channels=whole_state_channels,
        #                      out_channels=self.config.MODEL.ACTION_DIM,
        #                      command_channels=self.config.MODEL.COMMAND_DIM,
        #                      speed_as_input=self.config.MODEL.POLICY.SPEED_INPUT,
        #                      )

    def calculate_geometry(self):
        """ Intrinsics and extrinsics for a single camera.
        See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
        and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
        """
        x, y, z = 1.3, 0.0, 1.3
        pitch, yaw = 0.0, 0.0
        fov = self.config.IMAGE.FOV
        h, w = self.config.IMAGE.DIM

        f = w / (2 * np.tan(fov * np.pi / 360))
        cx = w / 2
        cy = h / 2
        intrinsics = np.float32([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]])

        # coordinate transform.
        P = np.float32([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
        extrinsics = self.get_extrinsics(x, y, z, yaw, pitch, 0) @ P

        return intrinsics, extrinsics

    @staticmethod
    def get_extrinsics(x, y, z, yaw, pitch, roll):
        a = np.radians(yaw)
        b = np.radians(pitch)
        g = np.radians(roll)

        c_a, c_b, c_g = np.cos(a), np.cos(b), np.cos(g)
        s_a, s_b, s_g = np.sin(a), np.sin(b), np.sin(g)

        mat = np.float32([
            [c_a * c_b, c_a * s_b * s_g - s_a * c_g, c_a * s_b * c_g + s_a * s_g, x],
            [s_a * c_b, s_a * s_b * s_g + c_a * c_g, s_a * s_b * c_g - c_a * s_g, y],
            [-s_b, c_b * s_g, c_b * c_g, z],
            [0, 0, 0, 1],
        ])

        mat[1, :] *= -1.0
        mat[:, 1] *= -1.0

        return mat

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.config.IMAGE.DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.config.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, batch, is_train=True):
        """
        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    'bev' (b, s, c, h, w)
                    'route_command' (b, s, c_route)
                    'speed' (b, s, 1)
        """
        # Encoder
        # Lifting features and project to bird's-eye view
        # (B, S, N, 3, 3) and (B, S, N, 4, 4)
        b, s, n = batch['image'].shape[:3]
        intrinsics = torch.FloatTensor(self.intrinsics).to(batch['image'].device)
        intrinsics = intrinsics.view(1, 1, 1, 3, 3).expand(b, s, n, 3, 3)
        extrinsics = torch.FloatTensor(self.extrinsics).to(batch['image'].device)
        extrinsics = extrinsics.view(1, 1, 1, 4, 4).expand(b, s, n, 4, 4)

        encoded_inputs = self.calculate_birds_eye_view_features(
            batch['image'], intrinsics, extrinsics
        )

        h, sample, z_mu, z_sigma, z_hat_mu, z_hat_sigma = self.rssm(
            input_embedding=encoded_inputs, action=batch['action'], is_train=is_train
        )

        output = {
            'future_mu': z_mu,
            'future_log_sigma': z_sigma,
            'present_mu': z_hat_mu,
            'present_log_sigma': z_hat_sigma,
        }

        reconstruction = self.decoder(torch.cat([h, sample], dim=-3))

        output['reconstruction'] = reconstruction

        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics):
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        x = self.encoder_forward(x)
        x = self.projection_to_birds_eye_view(x, geometry)
        x = self.downsample_module(x)
        x = unpack_sequence_dim(x, b, s)
        return x

    def encoder_forward(self, x):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def projection_to_birds_eye_view(self, x, geometry):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points
