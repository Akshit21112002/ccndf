import click
from os.path import join, dirname, abspath
import subprocess
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml
from pytorch_lightning import Callback
import matplotlib.pyplot as plt

import loc_ndf.datasets.datasets as datasets
import loc_ndf.models.models as models
from loc_ndf.utils import utils

class LossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss_values = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Print available keys in callback_metrics
        #print("Available keys:", trainer.callback_metrics.keys())
        
        # Append the training loss to train_loss_values
        if 'train/loss' in trainer.callback_metrics:
            #print("yes")
            self.train_loss_values.append(trainer.callback_metrics['train/loss'].item())
        else:
            print("Warning: 'train_loss' not found in callback_metrics.")


@click.command()
# Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(utils.CONFIG_DIR, 'config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
def main(config, weights, checkpoint):
    cfg = yaml.safe_load(open(config))
    cfg['git_commit_version'] = str(subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).strip())

    # Load data and model
    data = datasets.DataModule(cfg)

    cfg['bounding_box'] = data.get_train_set().bounding_box
    print(cfg['bounding_box'])

    if weights is None:
        model = models.LocNDF(cfg)
    else:
        model = models.LocNDF.load_from_checkpoint(
            weights, strict=False, hparams=cfg)
    model.update_map_params(data.get_train_set().points)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_dir = '/home/cvlab/LocNDF/exp'
    checkpoint_saver = ModelCheckpoint(monitor='train/loss',
                                       filename='best',
                                       mode='min',
                                       dirpath=checkpoint_dir,
                                       save_last=True)

    tb_logger = pl_loggers.TensorBoardLogger(join(utils.EXPERIMENT_DIR, cfg['experiment']['id']),
                                             default_hp_metric=False)

    loss_logger = LossLogger()

    trainer = Trainer(accelerator='gpu',
                      devices=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      callbacks=[lr_monitor, checkpoint_saver, loss_logger],
                      max_epochs=cfg['train']['max_epoch'])

    # Train!
    trainer.fit(model, data)

    # Plot and save the training loss curve
    print("len ",np.array(loss_logger.train_loss_values).shape)
    plt.plot(loss_logger.train_loss_values)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('/home/cvlab/LocNDF/loss_curve/loss_curve.jpg')

if __name__ == "__main__":
    main()
