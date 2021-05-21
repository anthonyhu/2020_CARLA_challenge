import os
import time
import socket

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from world_model.config import get_parser, get_cfg
from world_model.trainer import WorldModelTrainer


def main():
    args = get_parser().parse_args()
    config = get_cfg(args)

    model = WorldModelTrainer(config.convert_to_dict())

    save_dir = os.path.join(
        config.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + config.TAG
    )
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    model_checkpoint = ModelCheckpoint(save_dir, save_last=True)

    # try:
    #     resume_from_checkpoint = sorted(cfg.save_dir.glob('*.ckpt'))[-1]
    # except:
    #     resume_from_checkpoint = None

    trainer = pl.Trainer(
        gpus=config.GPUS,
        accelerator='ddp',
        precision=config.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=config.GRAD_NORM_CLIP,
        max_epochs=config.EPOCHS,
        resume_from_checkpoint=None,
        logger=logger,
        callbacks=model_checkpoint,
        log_every_n_steps=config.LOGGING_INTERVAL,
        plugins=DDPPlugin(find_unused_parameters=True),
        profiler='simple',
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
