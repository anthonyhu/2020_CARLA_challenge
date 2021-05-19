import os
import argparse
import pathlib
import time
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin


from pytorch_lightning.loggers import WandbLogger
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from carla_project.src.dataset import get_dataset_sequential


class Policy(nn.Module):
    def __init__(self, in_channels=64, out_channels=4, name='efficientnet-b0'):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(name)
        self.backbone._conv_stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, bias=False, padding=1)

        self.output_net = nn.Sequential(nn.Conv2d(320, 512, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(512, momentum=0.01),
                                        self.backbone._swish,
                                        nn.AdaptiveAvgPool2d(output_size=1),
                                        nn.Flatten(),
                                        nn.Linear(512, out_channels),
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
        x = self.get_features(x)  # get feature vector
        x = self.output_net(x)
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


class SegmentationLoss(nn.Module):
    def __init__(self, use_top_k=False, top_k_ratio=1.0):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
        )

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)


class ActionLoss(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-1, keepdims=True)
        return loss.mean()


class WorldModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.config = hparams
        self.policy = Policy(in_channels=9, out_channels=2)

        if self.config.use_transition:
            print('Enabled: Next state prediction')
            self.transition_model = TransitionModel(in_channels=9, action_channels=2)
            self.segmentation_loss = SegmentationLoss(use_top_k=True, top_k_ratio=0.5)
        self.policy_loss = ActionLoss(norm=1)

        #if self.config.use_reward:
            #print('Enabled: Reward')
         #   self.reward_model = RewardModel(in_channels=9, trajectory_length=self.config.sequence_length, n_actions=2)

    def forward(self, batch):
        state = batch['bev']
        b, s, c, h, w = state.shape

        input_policy = state.view(b*s, c, h, w)
        predicted_actions = self.policy(input_policy)
        predicted_actions = predicted_actions.view(b, s, -1)

        predicted_states = None
        if self.config.use_transition:
            input_transition_states = state[:, :-1].contiguous().view(b * (s - 1), c, h, w)
            input_transition_actions = batch['action'][:, :-1].contiguous().view(b * (s - 1), -1)
            predicted_states = self.transition_model(input_transition_states, input_transition_actions)
            predicted_states = predicted_states.view(b, s-1, c, h, w)

        return predicted_actions, predicted_states

    def shared_step(self, batch, is_train=False):
        predicted_actions, predicted_states = self.forward(batch)

        action_loss = self.policy_loss(predicted_actions, batch['action'])

        future_prediction_loss = action_loss.new_zeros(1)
        if self.config.use_transition:
            target_states = torch.argmax(batch['bev'][:, 1:], dim=-3)
            future_prediction_loss = self.segmentation_loss(predicted_states, target_states)

        return {'future_prediction': future_prediction_loss,
                'action': action_loss
                }

    def training_step(self, batch, batch_nb):
        loss = self.shared_step(batch, is_train=True)

        return {'loss': sum(loss.values())}

    def validation_step(self, batch, batch_nb):
        loss = self.shared_step(batch, is_train=False)

        return {'val_loss': sum(loss.values()).item()}

    def configure_optimizers(self):
        params = list(self.policy.parameters())

        if self.config.use_transition:
            params = params + list(self.transition_model.parameters())
        if self.config.use_reward:
            params = params + list(self.reward_model.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        return optimizer

    def train_dataloader(self):
        return get_dataset_sequential(
            pathlib.Path(self.config.dataset_dir), is_train=True, batch_size=self.config.batch_size,
            num_workers=self.config.num_workers, sequence_length=self.config.sequence_length,
        )

    def val_dataloader(self):
        return get_dataset_sequential(
            pathlib.Path(self.config.dataset_dir), is_train=False, batch_size=self.config.batch_size,
            num_workers=self.config.num_workers, sequence_length=self.config.sequence_length,
        )


def main(config):
    model = WorldModel(config)

    save_dir = os.path.join(
        config.save_dir, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + config.id
    )
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)

    # try:
    #     resume_from_checkpoint = sorted(config.save_dir.glob('*.ckpt'))[-1]
    # except:
    #     resume_from_checkpoint = None

    trainer = pl.Trainer(
        gpus=-1,
        accelerator='ddp',
        sync_batchnorm=True,
        max_epochs=config.max_epochs,
        resume_from_checkpoint=None,
        logger=logger,
        plugins=DDPPlugin(find_unused_parameters=True),
        profiler='simple',
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='/data/cornucopia/ah2029/experiments/carla/transition_model')
    parser.add_argument('--id', type=str, default='debug')

    # Data args.
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model args
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--use_transition', type=bool, default=False)
    parser.add_argument('--use_reward', type=bool, default=False)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parsed = parser.parse_args()

    main(parsed)
