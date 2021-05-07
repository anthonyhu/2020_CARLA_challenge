import uuid
import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import wandb


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from carla_project.src.models import SegmentationModel, RawController
from carla_project.src.utils.heatmap import ToHeatmap
from carla_project.src.dataset import get_dataset
from carla_project.src.converter import Converter
from carla_project.src import common
from carla_project.src.scripts.cluster_points import points as RANDOM_POINTS


@torch.no_grad()
def viz(batch, out, out_ctrl, target_cam, lbl_cam, lbl_map, ctrl_map, point_loss, ctrl_loss):
    images = list()

    for i in range(out.shape[0]):
        _point_loss = point_loss[i]
        _ctrl_loss = ctrl_loss[i]

        _out = out[i]
        _target = target_cam[i]
        _lbl_cam = lbl_cam[i]
        _lbl_map = lbl_map[i]

        _out_ctrl = out_ctrl[i]
        _ctrl_map = ctrl_map[i]

        img, topdown, points, _, actions, meta = [x[i] for x in batch]

        _rgb = Image.fromarray((255 * img[:3].cpu()).byte().numpy().transpose(1, 2, 0))
        _draw_rgb = ImageDraw.Draw(_rgb)
        _draw_rgb.text((5, 10), 'Point loss: %.3f' % _point_loss)
        _draw_rgb.text((5, 30), 'Control loss: %.3f' % _ctrl_loss)
        _draw_rgb.text((5, 50), 'Raw: %.3f %.3f' % tuple(_out_ctrl))
        _draw_rgb.text((5, 70), 'Pred: %.3f %.3f' % tuple(_ctrl_map))
        _draw_rgb.text((5, 90), 'Meta: %s' % meta)
        _draw_rgb.ellipse((_target[0]-3, _target[1]-3, _target[0]+3, _target[1]+3), (255, 255, 255))

        for x, y in _out:
            x = (x + 1) / 2 * _rgb.width
            y = (y + 1) / 2 * _rgb.height

            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 255, 0))

        for x, y in _lbl_cam:
            x = (x + 1) / 2 * _rgb.width
            y = (y + 1) / 2 * _rgb.height

            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).cpu().numpy()])
        _draw_map = ImageDraw.Draw(_topdown)

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw_map.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        for x, y in _lbl_map:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown.thumbnail(_rgb.size)

        image = np.hstack((_rgb, _topdown)).transpose(2, 0, 1)
        images.append((_ctrl_loss, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result


class Policy(nn.Module):
    def __init__(self, in_channels=64, out_channels=4, name='efficientnet-b0'):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(name)
        self.backbone._conv_stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, bias=False, padding=1)

        self.control_net = nn.Sequential(nn.Conv2d(320, 512, kernel_size=1, bias=False),
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
        x = self.control_net(x)
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


class ModelBasedNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)

        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.converter = Converter()
        self.controller = RawController(4)

    def forward(self, img, target):
        target_cam = self.converter.map_to_cam(target)
        target_heatmap_cam = self.to_heatmap(target, img)[:, None]
        out = self.net(torch.cat((img, target_heatmap_cam), 1))

        return out, target_cam, target_heatmap_cam

    @torch.no_grad()
    def _get_labels(self, topdown, target):
        out, (target_heatmap,) = self.teacher.forward(topdown, target, debug=True)
        control = self.teacher.controller(out)

        return out, control, (target_heatmap,)

    def training_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta = batch

        # Ground truth command.
        lbl_map, ctrl_map, (target_heatmap,) = self._get_labels(topdown, target)
        lbl_cam = self.converter.map_to_cam((lbl_map + 1) / 2 * 256)
        lbl_cam[..., 0] = (lbl_cam[..., 0] / 256) * 2 - 1
        lbl_cam[..., 1] = (lbl_cam[..., 1] / 144) * 2 - 1

        out, target_cam, target_heatmap_cam = self.forward(img, target)

        alpha = torch.rand(out.shape[0], out.shape[1], 1).type_as(out)
        between = alpha * out + (1-alpha) * lbl_cam
        out_ctrl = self.controller(between)

        point_loss = torch.nn.functional.l1_loss(out, lbl_cam, reduction='none').mean((1, 2))
        ctrl_loss_raw = torch.nn.functional.l1_loss(out_ctrl, ctrl_map, reduction='none')
        ctrl_loss = ctrl_loss_raw.mean(1)
        steer_loss = ctrl_loss_raw[:, 0]
        speed_loss = ctrl_loss_raw[:, 1]

        loss_gt = (point_loss + self.hparams.command_coefficient * ctrl_loss)
        loss_gt_mean = loss_gt.mean()

        # Random command.
        indices = np.random.choice(RANDOM_POINTS.shape[0], topdown.shape[0])
        target_aug = torch.from_numpy(RANDOM_POINTS[indices]).type_as(img)

        lbl_map_aug, ctrl_map_aug, (target_heatmap_aug,) = self._get_labels(topdown, target_aug)
        lbl_cam_aug = self.converter.map_to_cam((lbl_map_aug + 1) / 2 * 256)
        lbl_cam_aug[..., 0] = (lbl_cam_aug[..., 0] / 256) * 2 - 1
        lbl_cam_aug[..., 1] = (lbl_cam_aug[..., 1] / 144) * 2 - 1

        out_aug, target_cam_aug, target_heatmap_cam_aug = self.forward(img, target_aug)

        alpha = torch.rand(out.shape[0], out.shape[1], 1).type_as(out)
        between_aug = alpha * out_aug + (1-alpha) * lbl_cam_aug
        out_ctrl_aug = self.controller(between_aug)

        point_loss_aug = torch.nn.functional.l1_loss(out_aug, lbl_cam_aug, reduction='none').mean((1, 2))
        ctrl_loss_aug_raw = torch.nn.functional.l1_loss(out_ctrl_aug, ctrl_map_aug, reduction='none')
        ctrl_loss_aug = ctrl_loss_aug_raw.mean(1)
        steer_loss_aug = ctrl_loss_aug_raw[:, 0]
        speed_loss_aug = ctrl_loss_aug_raw[:, 1]

        loss_aug = (point_loss_aug + self.hparams.command_coefficient * ctrl_loss_aug)
        loss_aug_mean = loss_aug.mean()

        loss = loss_gt_mean + loss_aug_mean
        metrics = {
                'train_loss': loss.item(),

                'train_point': point_loss.mean().item(),
                'train_ctrl': ctrl_loss.mean().item(),
                'train_steer': steer_loss.mean().item(),
                'train_speed': speed_loss.mean().item(),

                'train_point_aug': point_loss_aug.mean().item(),
                'train_ctrl_aug': ctrl_loss_aug.mean().item(),
                'train_steer_aug': steer_loss_aug.mean().item(),
                'train_speed_aug': speed_loss_aug.mean().item(),
                }

        if batch_nb % 250 == 0:
            metrics['train_image'] = viz(batch, out, out_ctrl, target_cam, lbl_cam, lbl_map, ctrl_map, point_loss, ctrl_loss)
            metrics['train_image_aug'] = viz(batch, out_aug, out_ctrl_aug, target_cam_aug,
                                             lbl_cam_aug, lbl_map_aug, ctrl_map_aug,
                                             point_loss_aug, ctrl_loss_aug)

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta = batch

        # Ground truth command.
        lbl_map, ctrl_map, (target_heatmap,) = self._get_labels(topdown, target)
        lbl_cam = self.converter.map_to_cam((lbl_map + 1) / 2 * 256)
        lbl_cam[..., 0] = (lbl_cam[..., 0] / 256) * 2 - 1
        lbl_cam[..., 1] = (lbl_cam[..., 1] / 144) * 2 - 1

        out, target_cam, target_heatmap_cam = self.forward(img, target)
        out_ctrl = self.controller(out)
        out_ctrl_gt = self.controller(lbl_cam)

        point_loss = torch.nn.functional.l1_loss(out, lbl_cam, reduction='none').mean((1, 2))
        ctrl_loss_raw = torch.nn.functional.l1_loss(out_ctrl, ctrl_map, reduction='none')
        ctrl_loss = ctrl_loss_raw.mean(1)
        steer_loss = ctrl_loss_raw[:, 0]
        speed_loss = ctrl_loss_raw[:, 1]

        ctrl_loss_gt_raw = torch.nn.functional.l1_loss(out_ctrl_gt, ctrl_map, reduction='none')
        ctrl_loss_gt = ctrl_loss_gt_raw.mean(1)
        steer_loss_gt = ctrl_loss_gt_raw[:, 0]
        speed_loss_gt = ctrl_loss_gt_raw[:, 1]

        loss_gt = (point_loss + self.hparams.command_coefficient * ctrl_loss)
        loss_gt_mean = loss_gt.mean()

        # Random command.
        indices = np.random.choice(RANDOM_POINTS.shape[0], topdown.shape[0])
        target_aug = torch.from_numpy(RANDOM_POINTS[indices]).type_as(img)

        lbl_map_aug, ctrl_map_aug, (target_heatmap_aug,) = self._get_labels(topdown, target_aug)
        lbl_cam_aug = self.converter.map_to_cam((lbl_map_aug + 1) / 2 * 256)
        lbl_cam_aug[..., 0] = (lbl_cam_aug[..., 0] / 256) * 2 - 1
        lbl_cam_aug[..., 1] = (lbl_cam_aug[..., 1] / 144) * 2 - 1
        out_aug, target_cam_aug, target_heatmap_cam_aug = self.forward(img, target_aug)
        out_ctrl_aug = self.controller(out_aug)
        out_ctrl_gt_aug = self.controller(lbl_cam_aug)

        point_loss_aug = torch.nn.functional.l1_loss(out_aug, lbl_cam_aug, reduction='none').mean((1, 2))

        ctrl_loss_aug_raw = torch.nn.functional.l1_loss(out_ctrl_aug, ctrl_map_aug, reduction='none')
        ctrl_loss_aug = ctrl_loss_aug_raw.mean(1)
        steer_loss_aug = ctrl_loss_aug_raw[:, 0]
        speed_loss_aug = ctrl_loss_aug_raw[:, 1]

        ctrl_loss_gt_aug_raw = torch.nn.functional.l1_loss(out_ctrl_gt_aug, ctrl_map_aug, reduction='none')
        ctrl_loss_gt_aug = ctrl_loss_gt_aug_raw.mean(1)
        steer_loss_gt_aug = ctrl_loss_gt_aug_raw[:, 0]
        speed_loss_gt_aug = ctrl_loss_gt_aug_raw[:, 1]

        loss_gt_aug = (point_loss_aug + self.hparams.command_coefficient * ctrl_loss_aug)
        loss_gt_aug_mean = loss_gt_aug.mean()

        if batch_nb == 0:
            self.logger.log_metrics({
                'val_image': viz(batch, out, out_ctrl, target_cam, lbl_cam, lbl_map, ctrl_map, point_loss, ctrl_loss),
                'val_image_aug': viz(batch, out_aug, out_ctrl_aug, target_cam_aug,
                                     lbl_cam_aug, lbl_map_aug, ctrl_map_aug,
                                     point_loss_aug, ctrl_loss_aug)
                }, self.global_step)

        return {
                'val_loss': (loss_gt_mean + loss_gt_aug_mean).item(),

                'val_point': point_loss.mean().item(),
                'val_ctrl': ctrl_loss.mean().item(),
                'val_steer': steer_loss.mean().item(),
                'val_speed': speed_loss.mean().item(),
                'val_ctrl_gt': ctrl_loss_gt.mean().item(),
                'val_steer_gt': steer_loss_gt.mean().item(),
                'val_speed_gt': speed_loss_gt.mean().item(),

                'val_point_aug': point_loss_aug.mean().item(),
                'val_ctrl_aug': ctrl_loss_aug.mean().item(),
                'val_steer_aug': steer_loss_aug.mean().item(),
                'val_speed_aug': speed_loss_aug.mean().item(),
                'val_ctrl_gt_aug': ctrl_loss_gt_aug.mean().item(),
                'val_steer_gt_aug': steer_loss_gt_aug.mean().item(),
                'val_speed_gt_aug': speed_loss_gt_aug.mean().item(),
                }

    def validation_epoch_end(self, outputs):
        results = dict()

        for output in outputs:
            for key in output:
                if key not in results:
                    results[key] = list()

                results[key].append(output[key])

        summary = {key: np.mean(val) for key, val in results.items()}
        self.logger.log_metrics(summary, self.global_step)

        return summary

    def configure_optimizers(self):
        optim = torch.optim.Adam(
                list(self.net.parameters()) + list(self.controller.parameters()),
                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=0.5, patience=5, min_lr=1e-6,
                verbose=True)

        return [optim], [scheduler]

    def train_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by)

    def val_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by)

    def state_dict(self):
        return {k: v for k, v in super().state_dict().items() if 'teacher' not in k}

    def load_state_dict(self, state_dict):
        errors = super().load_state_dict(state_dict, strict=False)

        print(errors)


def main(hparams):
    try:
        resume_from_checkpoint = sorted(hparams.save_dir.glob('*.ckpt'))[-1]
    except:
        resume_from_checkpoint = None

    model = ModelBasedNet(hparams)
    logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='stage_2')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=1)

    trainer = pl.Trainer(
            gpus=-1, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

    wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    state_n_channels = 64
    policy = Policy(in_channels=state_n_channels)

    x = torch.zeros((1, state_n_channels, 512, 512))
    action = torch.arange(4).float().view(1, 4)
    policy(x)

    transition_model = TransitionModel(in_channels=state_n_channels)

    x2 = transition_model(x, action)




    # parser = argparse.ArgumentParser()
    # parser.add_argument('--max_epochs', type=int, default=50)
    # parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    # parser.add_argument('--id', type=str, default=uuid.uuid4().hex)
    #
    # parser.add_argument('--teacher_path', type=pathlib.Path, required=True)
    #
    # # Model args.
    # parser.add_argument('--heatmap_radius', type=int, default=5)
    # parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    # parser.add_argument('--command_coefficient', type=float, default=0.1)
    # parser.add_argument('--temperature', type=float, default=5.0)
    # parser.add_argument('--hack', action='store_true', default=False)
    #
    # # Data args.
    # parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    # parser.add_argument('--batch_size', type=int, default=16)
    #
    # # Optimizer args.
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--weight_decay', type=float, default=0.0)
    #
    # parsed = parser.parse_args()
    # parsed.teacher_path = parsed.teacher_path.resolve()
    # parsed.save_dir = parsed.save_dir.resolve() / parsed.id
    # parsed.save_dir.mkdir(parents=True, exist_ok=True)
    #
    # main(parsed)
