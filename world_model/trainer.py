import os
import pathlib

import torch
import pytorch_lightning as pl

from world_model.config import get_cfg
from world_model.dataset import get_dataset_sequential
from world_model.models import Policy, TransitionModel, RewardModel
from world_model.losses import ActionLoss, SegmentationLoss
from world_model.utils import set_bn_momentum


class WorldModelTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.config = get_cfg(cfg_dict=hparams)

        # Dataset
        self.dataset_path = os.path.join(self.config.DATASET.DATAROOT, self.config.DATASET.VERSION)

        # Model
        self.policy = Policy(in_channels=self.config.MODEL.IN_CHANNELS, out_channels=self.config.MODEL.ACTION_DIM,
                             command_channels=self.config.MODEL.COMMAND_DIM
                             )

        if self.config.MODEL.TRANSITION.ENABLED:
            print('Enabled: Next state prediction')
            self.transition_model = TransitionModel(
                in_channels=self.config.MODEL.IN_CHANNELS, action_channels=self.config.MODEL.ACTION_DIM,
            )
            self.segmentation_loss = SegmentationLoss(
                use_top_k=self.config.SEMANTIC_SEG.USE_TOP_K, top_k_ratio=self.config.SEMANTIC_SEG.TOP_K_RATIO,
            )
        self.policy_loss = ActionLoss(norm=1)

        if self.config.MODEL.REWARD.ENABLED:
            print('Enabled: Reward')
            self.reward_model = RewardModel(
                in_channels=self.config.MODEL.IN_CHANNELS,
                trajectory_length=self.config.SEQUENCE_LENGTH,
                n_actions=self.config.MODEL.ACTION_DIM,
            )

        set_bn_momentum(self, self.config.MODEL.BN_MOMENTUM)

    def forward(self, batch):
        state = batch['bev']
        b, s, c, h, w = state.shape

        input_policy = state.view(b*s, c, h, w)

        route_command = batch['route_command'].view(b * s, -1)
        predicted_actions = self.policy(input_policy, route_command)
        predicted_actions = predicted_actions.view(b, s, -1)

        predicted_states = None
        if self.config.MODEL.TRANSITION.ENABLED:
            input_transition_states = state[:, :-1].contiguous().view(b * (s - 1), c, h, w)
            input_transition_actions = batch['action'][:, :-1].contiguous().view(b * (s - 1), -1)
            predicted_states = self.transition_model(input_transition_states, input_transition_actions)
            predicted_states = predicted_states.view(b, s-1, c, h, w)

        return predicted_actions, predicted_states

    def shared_step(self, batch, is_train=False):
        predicted_actions, predicted_states = self.forward(batch)

        action_loss = self.policy_loss(predicted_actions, batch['action'])

        future_prediction_loss = action_loss.new_zeros(1)
        if self.config.MODEL.TRANSITION.ENABLED:
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

        if self.config.MODEL.TRANSITION.ENABLED:
            params = params + list(self.transition_model.parameters())
        if self.config.MODEL.REWARD.ENABLED:
            params = params + list(self.reward_model.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.config.OPTIMIZER.LR, weight_decay=self.config.OPTIMIZER.WEIGHT_DECAY,
        )

        return optimizer

    def train_dataloader(self):
        return get_dataset_sequential(
            pathlib.Path(self.dataset_path), is_train=True, batch_size=self.config.BATCHSIZE,
            num_workers=self.config.N_WORKERS, sequence_length=self.config.SEQUENCE_LENGTH,
        )

    def val_dataloader(self):
        return get_dataset_sequential(
            pathlib.Path(self.dataset_path), is_train=False, batch_size=self.config.BATCHSIZE,
            num_workers=self.config.N_WORKERS, sequence_length=self.config.SEQUENCE_LENGTH,
        )