import os
import pathlib

import torch
import pytorch_lightning as pl

from world_model.config import get_cfg
from world_model.dataset import get_dataset_sequential
from world_model.models import TemporalModel, TemporalModelIdentity, Encoder, Policy, TransitionModel, RewardModel, \
    Distribution
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

        # Temporal model
        self.receptive_field = self.config.RECEPTIVE_FIELD
        if self.receptive_field == 1:
            self.temporal_model = TemporalModelIdentity(
                in_channels=self.config.MODEL.ENCODER.OUTPUT_DIM, receptive_field=self.receptive_field,
            )
        else:
            self.temporal_model = TemporalModel(
                in_channels=self.config.MODEL.ENCODER.OUTPUT_DIM, receptive_field=self.receptive_field,
                input_shape=(8, 8), start_out_channels=self.config.MODEL.TEMPORAL_MODEL.OUTPUT_DIM,
            )

        state_channels = self.temporal_model.out_channels
        whole_state_channels = state_channels
        if self.config.MODEL.PROBABILISTIC.ENABLED:
            # Input: s_t
            self.present_distribution = Distribution(
                in_channels=state_channels, latent_dim=self.config.MODEL.PROBABILISTIC.LATENT_DIM
            )

            # Input: [s_t, s_{t+1}]
            self.future_distribution = Distribution(
                in_channels=2*state_channels, latent_dim=self.config.MODEL.PROBABILISTIC.LATENT_DIM
            )

            whole_state_channels += self.config.MODEL.PROBABILISTIC.LATENT_DIM

            self.probabilistic_loss = ProbabilisticLoss()

        self.policy = Policy(in_channels=whole_state_channels,
                             out_channels=self.config.MODEL.ACTION_DIM,
                             command_channels=self.config.MODEL.COMMAND_DIM,
                             speed_as_input=self.config.MODEL.POLICY.SPEED_INPUT,
                             )

        if self.config.MODEL.TRANSITION.ENABLED:
            print('Enabled: Next state prediction')
            self.transition_model = TransitionModel(
                in_channels=whole_state_channels, action_channels=self.config.MODEL.ACTION_DIM,
                out_channels=self.config.MODEL.ENCODER.OUTPUT_DIM,
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
        _, enc_c, enc_h, enc_w = encoded_inputs.shape
        encoded_inputs = encoded_inputs.view(b, s, enc_c, enc_h, enc_w)
        # Temporal model
        latent_state = self.temporal_model(encoded_inputs)

        # Policy
        b, s, c, h, w = latent_state.shape
        route_command = batch['route_command'][:, (self.receptive_field - 1):self.receptive_field].contiguous().view(b, -1)
        speed = batch['speed'][:, (self.receptive_field - 1):self.receptive_field].contiguous().view(b, -1)

        distribution_output = None

        if not deployment and self.config.MODEL.TRANSITION.ENABLED:
            policy_input = latent_state[:, :-1].contiguous()
        else:
            policy_input = latent_state
        if self.config.MODEL.PROBABILISTIC.ENABLED:
            distribution_output = self.distribution_forward(latent_state, deployment=deployment)

            policy_input = self.add_stochastic_state(policy_input, distribution_output, deployment=deployment)

        predicted_actions = self.policy(policy_input[:, 0], route_command, speed)
        predicted_actions = predicted_actions.view(b, 1, -1)

        # Future prediction
        future_state = None
        if self.config.MODEL.TRANSITION.ENABLED:
            if deployment:
                future_state = torch.zeros_like(latent_state)
            else:
                input_transition_states = policy_input.view(b, -1, h, w)
                input_transition_actions = batch['action'][:, (self.receptive_field - 1):self.receptive_field].contiguous().view(b, -1)
                future_state = self.transition_model(input_transition_states, input_transition_actions)
                future_state = future_state.view(b, 1, -1, h, w)

        output = {'action': predicted_actions,
                  'future_state': future_state,
                  'latent_state': latent_state,
        }

        if self.config.MODEL.PROBABILISTIC.ENABLED:
            output = {**output, **distribution_output}

        return output

    def distribution_forward(self, latent_state, deployment=False):
        output = dict()
        output['present_mu'], output['present_log_sigma'] = self.present_distribution(latent_state[:, 0])

        if not deployment:
            assert latent_state.shape[1] == 2, 'Need two states. The sequence length is too short.'
            b, s, c, h, w = latent_state.shape
            output['future_mu'], output['future_log_sigma'] = self.future_distribution(latent_state.view(b, s*c, h, w))

        return output

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

        # Policy loss
        action_loss = self.policy_loss(
            output['action'], batch['action'][:, (self.receptive_field - 1):self.receptive_field]
        )

        # Future prediction loss
        future_prediction_loss = action_loss.new_zeros(1)
        if self.config.MODEL.TRANSITION.ENABLED:
            #target_states = torch.argmax(batch['bev'][:, 1:], dim=-3)
            future_prediction_loss = self.future_pred_loss(output['future_state'], output['latent_state'][:, 1:])

        probabilistic_loss = action_loss.new_zeros(1)
        if self.config.MODEL.PROBABILISTIC.ENABLED:
            probabilistic_loss = self.probabilistic_loss(output)

        losses = {'future_prediction': future_prediction_loss,
                  'action': action_loss,
                  'probabilistic': probabilistic_loss,
                  #'brake': brake_loss,
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
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

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
