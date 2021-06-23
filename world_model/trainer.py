import os
import pathlib

import torch
import pytorch_lightning as pl

from world_model.config import get_cfg
from world_model.dataset import get_dataset_sequential
from world_model.models import TemporalModel, TemporalModelIdentity, Encoder, Policy, TransitionModel, RewardModel
from world_model.losses import RegressionLoss, SegmentationLoss
from world_model.utils import set_bn_momentum


class WorldModelTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.config = get_cfg(cfg_dict=hparams)

        # Dataset
        self.dataset_path = os.path.join(self.config.DATASET.DATAROOT, self.config.DATASET.VERSION)

        # Model
        self.receptive_field = self.config.RECEPTIVE_FIELD
        if self.receptive_field == 1:
            self.temporal_model = TemporalModelIdentity(
                in_channels=self.config.MODEL.IN_CHANNELS, receptive_field=self.receptive_field,
            )
        else:
            self.temporal_model = TemporalModel(
                in_channels=self.config.MODEL.IN_CHANNELS, receptive_field=self.receptive_field,
                input_shape=self.config.IMAGE.DIM, start_out_channels=self.config.MODEL.TEMPORAL_MODEL.OUTPUT_DIM,
            )

        state_in_channel = self.temporal_model.out_channels

        self.encoder = Encoder(in_channels=state_in_channel,
                               out_channels=256,
                               )

        self.policy = Policy(in_channels=256,
                             out_channels=self.config.MODEL.ACTION_DIM,
                             command_channels=self.config.MODEL.COMMAND_DIM,
                             speed_as_input=self.config.MODEL.POLICY.SPEED_INPUT,
                             )

        if self.config.MODEL.TRANSITION.ENABLED:
            print('Enabled: Next state prediction')
            self.transition_model = TransitionModel(
                in_channels=256, action_channels=self.config.MODEL.ACTION_DIM,
            )
            # self.segmentation_loss = SegmentationLoss(
            #     use_top_k=self.config.SEMANTIC_SEG.USE_TOP_K, top_k_ratio=self.config.SEMANTIC_SEG.TOP_K_RATIO,
            # )
            self.future_pred_loss = RegressionLoss(norm=2, channel_dim=-3)

            assert self.receptive_field + 1 == self.config.SEQUENCE_LENGTH
        else:
            assert self.receptive_field == self.config.SEQUENCE_LENGTH

        self.policy_loss = RegressionLoss(norm=1, channel_dim=-1)

        if self.config.MODEL.REWARD.ENABLED:
            print('Enabled: Reward')
            self.reward_model = RewardModel(
                in_channels=self.config.MODEL.IN_CHANNELS,
                trajectory_length=self.config.SEQUENCE_LENGTH,
                n_actions=self.config.MODEL.ACTION_DIM,
            )

            self.adversarial_loss = torch.nn.MSELoss()

        set_bn_momentum(self, self.config.MODEL.BN_MOMENTUM)

    def forward(self, batch, deployment=False):
        """
        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    'bev' (b, s, c, h, w)
                    'route_command' (b, s, c_route)
                    'speed' (b, s, 1)
        """
        # Temporal model
        state = self.temporal_model(batch['bev'])

        # Policy
        b, s, c, h, w = state.shape
        route_command = batch['route_command'][:, (self.receptive_field - 1):].contiguous().view(b * s, -1)
        speed = batch['speed'][:, (self.receptive_field - 1):].contiguous().view(b * s, -1)
        input_policy = state.view(b*s, c, h, w)

        latent_state = self.encoder(input_policy)
        predicted_actions = self.policy(latent_state, route_command, speed)
        predicted_actions = predicted_actions.view(b, s, -1)

        # Future prediction
        future_state = None
        if self.config.MODEL.TRANSITION.ENABLED:
            if deployment:
                # Predict next future state
                input_transition_states = state[:, -1]
                input_transition_actions = predicted_actions[:, -1]
                future_state = self.transition_model(input_transition_states, input_transition_actions)
                future_state = future_state.view(b, 1, c, h, w)
            else:
                _, new_c, new_h, new_w = latent_state.shape
                latent_state = latent_state.view(b, s, new_c, new_h, new_w)
                input_transition_states = latent_state[:, :-1].contiguous().view(b * (s - 1), new_c, new_h, new_w)
                input_transition_actions = batch['action'][:, :-1].contiguous().view(b * (s - 1), -1)
                future_state = self.transition_model(input_transition_states, input_transition_actions)
                future_state = future_state.view(b, s-1, new_c, new_h, new_w)

        return predicted_actions, future_state, latent_state

    def shared_step(self, batch, is_train, optimizer_idx=0):
        predicted_actions, future_state, latent_state = self.forward(batch)

        # Policy loss
        action_loss = self.policy_loss(predicted_actions, batch['action'][:, (self.receptive_field - 1):])

        # Future prediction loss
        future_prediction_loss = action_loss.new_zeros(1)
        if self.config.MODEL.TRANSITION.ENABLED:
            #target_states = torch.argmax(batch['bev'][:, 1:], dim=-3)
            future_prediction_loss = self.future_pred_loss(future_state, latent_state[:, 1:])

        losses = {'future_prediction': future_prediction_loss,
                  'action': action_loss,
                  #'brake': brake_loss,
                  }

        if not self.config.MODEL.REWARD.ENABLED:
            return losses

        # Train generator and discriminator
        # See blog post https://towardsdatascience.com/how-to-train-a-gan-on-128-gpus-using-pytorch-9a5b27a52c73
        if optimizer_idx == 0:
            valid = torch.ones(predicted_actions.size(0), 1)
            valid = valid.type_as(predicted_actions)

            # adversarial loss is binary cross-entropy
            reward_loss = self.adversarial_loss(self.predicted_actions, valid)

        if optimizer_idx == 1:
            valid = torch.ones(predicted_actions.size(0), 1)
            valid = valid.type_as(predicted_actions)

            valid_loss = self.adversarial_loss(self.predicted_actions.detach(), valid)

            fake = torch.zeros(predicted_actions.size(0), 1)
            fake = fake.type_as(predicted_actions)

            fake_loss = self.adversarial_loss(batch['action'], fake)

            reward_loss = (valid_loss + fake_loss) / 2

        losses['reward_loss'] = reward_loss

        return losses

    #def training_step(self, batch, batch_idx, optimizer_idx):
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, is_train=True, optimizer_idx=0)

        return {'loss': sum(loss.values())}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, is_train=False)

        return {'val_loss': sum(loss.values()).item()}

    def configure_optimizers(self):
        params = list(self.policy.parameters())

        if self.config.MODEL.TRANSITION.ENABLED:
            params = params + list(self.transition_model.parameters())

        optimizer = torch.optim.Adam(
            params, lr=self.config.OPTIMIZER.LR, weight_decay=self.config.OPTIMIZER.WEIGHT_DECAY,
        )

        if self.config.MODEL.REWARD.ENABLED:
            discriminator_optimizer = torch.optim.Adam(
                self.reward_model.parameters(),
                lr=self.config.OPTIMIZER.LR,
                weight_decay=self.config.OPTIMIZER.WEIGHT_DECAY,
            )
            return optimizer, discriminator_optimizer
        else:
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
