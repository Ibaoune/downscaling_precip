from dataset import DownscalingDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch
from models.losses import BernoulliGammaLoss
import matplotlib.pyplot as plt
import cmocean as cmo
from plot import spatial_comparison_per_epoch
import numpy as np

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict,
        batch_size: int = 1,
        num_workers: int = 0
    ):
        super().__init__()
        self.data_config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        if not hasattr(self, 'train_ds'):
            kwargs = self.data_config['common_kwargs'].copy()
            # update with train-specific args
            kwargs.update(self.data_config["train"])
            self.train_ds = DownscalingDataset(**kwargs)
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if not hasattr(self, 'val_ds'):
            kwargs = self.data_config['common_kwargs'].copy()
            kwargs.update(self.data_config["val"])
            self.val_ds = DownscalingDataset(**kwargs)
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        if not hasattr(self, 'test_ds'):
            kwargs = self.data_config['common_kwargs'].copy()
            kwargs.update(self.data_config["test"])
            self.test_ds = DownscalingDataset(**kwargs)
        return DataLoader(
            self.test_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

class LitModule(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, learning_rate):
        super(LitModule, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        # add metrics here if needed
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('test_loss', loss)
        # add metrics here if needed
        return loss
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        # skip if sanity check (an empty run at the start of training)
        if self.trainer.sanity_checking:
            return
        # plot one example from the validation set
        sample_batch = next(iter(self.trainer.val_dataloaders))
        x, y_true = sample_batch
        y_hat = self(x.to(self.device))
        if isinstance(self.criterion, BernoulliGammaLoss):
            # Expected value of Bernoulli-Gamma
            y_hat = y_hat[:, 0:1, :, :] * y_hat[:, 1:2, :, :] * y_hat[:, 2:3, :, :]
        
        y_pred_denorm = self.trainer.train_dataloader.dataset.denormalize(y_hat, data_type="y")
        y_true_denorm = self.trainer.train_dataloader.dataset.denormalize(y_true, data_type="y")
        
        fig = spatial_comparison_per_epoch(
            y_true_denorm[0].cpu().numpy(),
            y_pred_denorm[0, 0].cpu().numpy(),
            extent=self.trainer.train_dataloader.dataset.extent,
        )
        # log figure to tensorboard
        self.logger.experiment.add_figure(f'Validation sample', fig, self.current_epoch)
        plt.close()

    @torch.no_grad()
    def on_train_end(self):
        # loop over validation set and plot means
        val_loader = self.trainer.val_dataloaders
        all_y_true, all_y_pred = [], []
        for batch in val_loader:
            x, y = batch
            y_hat = self(x.to(self.device))
            if isinstance(self.criterion, BernoulliGammaLoss):
                # Expected value of Bernoulli-Gamma
                y_hat = y_hat[:, 0:1, :, :] * y_hat[:, 1:2, :, :] * y_hat[:, 2:3, :, :]
            y_pred_denorm = self.trainer.train_dataloader.dataset.denormalize(y_hat, data_type="y")
            y_true_denorm = self.trainer.train_dataloader.dataset.denormalize(y, data_type="y")
            all_y_true.append(y_true_denorm.cpu().numpy())
            all_y_pred.append(y_pred_denorm.cpu().numpy())
        mean_y_true = np.mean(np.concatenate(all_y_true, axis=0), axis=0)
        mean_y_pred = np.mean(np.concatenate(all_y_pred, axis=0), axis=0)[0]
        fig = spatial_comparison_per_epoch(
            mean_y_true,
            mean_y_pred,
            extent=self.trainer.train_dataloader.dataset.extent,
        )
        # log figure to tensorboard
        self.logger.experiment.add_figure(f'Validation Mean Comparison', fig, self.current_epoch)
        plt.close()

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = linear_warmup_scheduler(
            optimizer,
            (0.1 * self.trainer.max_epochs),
            self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
def linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) /
            float(max(1, total_steps - warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)
