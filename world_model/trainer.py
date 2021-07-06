import os
import pathlib

import torch
import torch.nn as nn
import pytorch_lightning as pl

from world_model.config import get_cfg
from world_model.dataset import get_dataset_sequential
from world_model.models import Encoder, RepresentationModel, RewardModel
from world_model.losses import RegressionLoss, SegmentationLoss, ProbabilisticLoss
from world_model.utils import set_bn_momentum


class WorldModelTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.config = get_cfg(cfg_dict=hparams)

        # Dataset
        self.dataset_path = os.path.join(self.config.DATASET.DATAROOT, self.config.DATASET.VERSION)

        #####
        # Model
        #####

        # Encoder
        self.encoder = Encoder(
            in_channels=self.config.MODEL.IN_CHANNELS, out_channels=self.config.MODEL.ENCODER.OUTPUT_DIM,
        )

        # Recurrent model
        self.receptive_field = self.config.RECEPTIVE_FIELD
        self.recurrent_model = nn.GRU(
            input_size=self.config.MODEL.ENCODER.OUTPUT_DIM + self.config.MODEL.ACTION_DIM,
            hidden_size=self.config.MODEL.RECURRENT_MODEL.OUTPUT_DIM,
            batch_first=True,
        )

        # Representation model
        state_channels = self.config.MODEL.RECURRENT_MODEL.OUTPUT_DIM
        self.representation_model = RepresentationModel(
            in_channels=state_channels + self.config.MODEL.ENCODER.OUTPUT_DIM,
            out_channels=2*state_channels,
        )

        self.predictor = RepresentationModel(in_channels=state_channels, out_channels=2*state_channels)

        self.probabilistic_loss = ProbabilisticLoss()

        # self.policy = Policy(in_channels=whole_state_channels,
        #                      out_channels=self.config.MODEL.ACTION_DIM,
        #                      command_channels=self.config.MODEL.COMMAND_DIM,
        #                      speed_as_input=self.config.MODEL.POLICY.SPEED_INPUT,
        #                      )

        if self.config.MODEL.TRANSITION.ENABLED:
            # print('Enabled: Next state prediction')
            # self.transition_model = TransitionModel(
            #     in_channels=whole_state_channels, action_channels=self.config.MODEL.ACTION_DIM,
            #     out_channels=self.config.MODEL.ENCODER.OUTPUT_DIM,
            # )
            # # self.segmentation_loss = SegmentationLoss(
            # #     use_top_k=self.config.SEMANTIC_SEG.USE_TOP_K, top_k_ratio=self.config.SEMANTIC_SEG.TOP_K_RATIO,
            # # )
            self.future_pred_loss = RegressionLoss(norm=2, channel_dim=-3)

        else:
            assert self.receptive_field == self.config.SEQUENCE_LENGTH

        #self.policy_loss = RegressionLoss(norm=1, channel_dim=-1)

        if self.config.MODEL.REWARD.ENABLED:
            print('Enabled: Reward')
            self.reward_model = RewardModel(
                in_channels=self.config.MODEL.IN_CHANNELS,
                trajectory_length=self.config.SEQUENCE_LENGTH,
                n_actions=self.config.MODEL.ACTION_DIM,
            )

            self.adversarial_loss = torch.nn.MSELoss()

        set_bn_momentum(self, self.config.MODEL.BN_MOMENTUM)

        self.training_step_count = 0

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
        # Encoder
        b, s, img_c, img_h, img_w = batch['bev'].shape
        encoded_inputs = self.encoder(batch['bev'].view(b*s, img_c, img_h, img_w))
        encoded_inputs = encoded_inputs.view(b, s, encoded_inputs.shape[-1])

        # Concatenate action to inputs.
        input_action = batch['action']
        encoded_inputs_action = torch.cat([encoded_inputs, input_action], dim=-1)

        # Recurrent model
        hidden_states, _ = self.recurrent_model(encoded_inputs_action)

        true_states = self.representation_model(torch.cat([hidden_states, encoded_inputs], dim=-1))
        predicted_states = self.predictor(hidden_states.contiguous())

        output = {
            'true_states': true_states,
            'predicted_states': predicted_states,
        }

        return output

    def compute_probabilistic_loss(self, output):
        true_mu, true_sigma = torch.split(output['true_states'], output['true_states'].shape[-1] // 2, dim=-1)
        pred_mu, pred_sigma = torch.split(output['predicted_states'], output['predicted_states'].shape[-1] // 2, dim=-1)

        input_loss = {
            'present_mu': pred_mu,
            'present_log_sigma': pred_sigma,
            'future_mu': true_mu,
            'future_log_sigma': true_sigma,
        }

        return self.probabilistic_loss(input_loss)

    def add_stochastic_state(self, policy_input, distribution_output, deployment):
        if not deployment:
            mu = distribution_output['future_mu']
            sigma = torch.exp(distribution_output['future_log_sigma'])
        else:
            mu = distribution_output['present_mu']
            sigma = torch.exp(distribution_output['present_log_sigma'])

        noise = torch.randn_like(mu)
        sample = mu + sigma * noise

        b, _, _, h, w = policy_input.shape
        latent_dim = sample.shape[1]
        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.view(b, 1, latent_dim, 1, 1).expand(b, 1, latent_dim, h, w)

        policy_input = torch.cat([policy_input, sample], dim=2)
        return policy_input

    def shared_step(self, batch, is_train, optimizer_idx=0):
        output = self.forward(batch)

        if self.config.MODEL.PROBABILISTIC.ENABLED:
            probabilistic_loss = self.compute_probabilistic_loss(output)

        losses = {
            'probabilistic': self.config.LOSSES.WEIGHT_PROBABILISTIC * probabilistic_loss,
        }

        if not self.config.MODEL.REWARD.ENABLED:
            return losses, output

        # Train generator and discriminator
        # See blog post https://towardsdatascience.com/how-to-train-a-gan-on-128-gpus-using-pytorch-9a5b27a52c73
        predicted_actions = output['action']
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

        return losses, output

    #def training_step(self, batch, batch_idx, optimizer_idx):
    def training_step(self, batch, batch_idx):
        loss, output = self.shared_step(batch, is_train=True, optimizer_idx=0)
        self.training_step_count += 1

        for key, value in loss.items():
            self.log('train_' + key, value)
        if self.training_step_count % self.config.VIS_INTERVAL == 0:
            pass
            #self.visualise(batch, output, batch_idx, prefix='train')

        return {'loss': sum(loss.values())}

    def validation_step(self, batch, batch_idx):
        loss, output = self.shared_step(batch, is_train=False)
        for key, value in loss.items():
            self.log('val_' + key, value)

        return {'val_loss': sum(loss.values()).item()}

    def visualise(self, labels, output, batch_idx, prefix='train'):
        # TODO
        pass
        # visualisation_video = visualise_output(labels, output, self.config)
        # name = f'{prefix}_outputs'
        # if prefix == 'val':
        #     name = name + f'_{batch_idx}'
        # self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

    def configure_optimizers(self):
        params = list(self.parameters())

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
