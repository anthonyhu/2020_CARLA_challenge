import os
import pathlib

import torch
import pytorch_lightning as pl

from world_model.config import get_cfg
from world_model.dataset import get_dataset_sequential
from world_model.models.fiery import Fiery
from world_model.losses import SegmentationLoss, KLBalancing
from world_model.metrics import IntersectionOverUnion
from world_model.utils import set_bn_momentum, COLOR


class WorldModelTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.config = get_cfg(cfg_dict=hparams)

        # Dataset
        self.dataset_path = os.path.join(self.config.DATASET.DATAROOT, self.config.DATASET.VERSION)

        # Model
        self.model = Fiery(self.config)

        # Losses
        self.probabilistic_loss = KLBalancing(alpha=self.config.LOSSES.KL_BALANCING_ALPHA)

        if self.config.MODEL.TRANSITION.ENABLED:
            self.segmentation_loss = SegmentationLoss(
                use_top_k=self.config.SEMANTIC_SEG.USE_TOP_K, top_k_ratio=self.config.SEMANTIC_SEG.TOP_K_RATIO,
            )

        else:
            assert self.model.receptive_field == self.config.SEQUENCE_LENGTH

        self.metric_iou_val = IntersectionOverUnion(n_classes=self.config.SEMANTIC_SEG.N_CHANNELS)

        #self.policy_loss = RegressionLoss(norm=1, channel_dim=-1)

        if self.config.MODEL.REWARD.ENABLED:
            print('Enabled: Reward')

            self.adversarial_loss = torch.nn.MSELoss()

        set_bn_momentum(self, self.config.MODEL.BN_MOMENTUM)

        self.training_step_count = 0

    def compute_probabilistic_loss(self, output):
        return self.probabilistic_loss(
            output['present_mu'], output['present_log_sigma'], output['future_mu'], output['future_log_sigma']
        )

    def shared_step(self, batch, is_train, optimizer_idx=0):
        output = self.model.forward(batch, is_train=is_train)

        reconstruction_loss = self.segmentation_loss(
            prediction=output['reconstruction'], target=torch.argmax(batch['bev'], dim=2)
        )

        if self.config.MODEL.PROBABILISTIC.ENABLED:
            probabilistic_loss = self.compute_probabilistic_loss(output)
        else:
            probabilistic_loss = torch.zeros_like(reconstruction_loss)

        losses = {
            'probabilistic': self.config.LOSSES.WEIGHT_PROBABILISTIC * probabilistic_loss,
            'reconstruction': self.config.LOSSES.WEIGHT_RECONSTRUCTION * reconstruction_loss,
        }

        if not is_train:
            seg_prediction = output['reconstruction'].detach()[:, self.model.receptive_field:]
            seg_prediction = torch.argmax(seg_prediction, dim=2)
            self.metric_iou_val(seg_prediction, torch.argmax(batch['bev'][:, self.model.receptive_field:], dim=2))

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

        self.metric_and_visualisation(batch, output, loss, batch_idx, prefix='train')

        return {'loss': self.loss_reducing(loss)}

    def validation_step(self, batch, batch_idx):
        loss, output = self.shared_step(batch, is_train=False)

        self.metric_and_visualisation(batch, output, loss, batch_idx, prefix='val')

        return {'val_loss': self.loss_reducing(loss).item()}

    def metric_and_visualisation(self, batch, output, loss, batch_idx, prefix='train'):
        for key, value in loss.items():
            for t, value_t in enumerate(value):
                self.log(f'{prefix}_{key}_time_{t}', value_t.item())

        if prefix == 'train':
            visualisation_criteria = self.training_step_count % self.config.VIS_INTERVAL == 0
        else:
            visualisation_criteria = batch_idx == 0
        if visualisation_criteria:
            self.visualise(batch, output, batch_idx, prefix=prefix)

    def loss_reducing(self, loss):
        total_loss = sum([x.mean() for x in loss.values()])
        return total_loss

    def shared_epoch_end(self, step_outputs, is_train):
        # log per class iou metrics
        class_names = ['unlabeled', 'pedestrian', 'road_line', 'road', 'sidewalk', 'vehicles', 'red_light',
                       'yellow_light', 'green_light']
        if not is_train:
            scores = self.metric_iou_val.compute()
            for key, value in zip(class_names, scores):
                self.logger.experiment.add_scalar('val_iou_' + key, value, global_step=self.training_step_count)
            self.logger.experiment.add_scalar('val_mean_iou', torch.mean(scores), global_step=self.training_step_count)
            self.metric_iou_val.reset()

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, True)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, False)

    def visualise(self, labels, output, batch_idx, prefix='train'):
        target = torch.argmax(labels['bev'], dim=2)
        pred = torch.argmax(output['reconstruction'], dim=2)

        colours = torch.tensor(COLOR, dtype=torch.uint8, device=pred.device)

        target = colours[target]
        pred = colours[pred]

        # Move channel to third position
        target = target.permute(0, 1, 4, 2, 3)
        pred = pred.permute(0, 1, 4, 2, 3)

        visualisation_video = torch.cat([target, pred], dim=-1).detach()

        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

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
            num_workers=self.config.N_WORKERS,
            debug_overfit=self.config.DEBUG_OVERFIT, sequence_length=self.config.SEQUENCE_LENGTH,
        )

    def val_dataloader(self):
        return get_dataset_sequential(
            pathlib.Path(self.dataset_path), is_train=False, batch_size=self.config.BATCHSIZE,
            num_workers=self.config.N_WORKERS,
            debug_overfit=self.config.DEBUG_OVERFIT, sequence_length=self.config.SEQUENCE_LENGTH,
        )
