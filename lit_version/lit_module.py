from dataset import DownscalingDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

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
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('val_loss', loss)
        # add metrics here if needed
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('test_loss', loss)
        # add metrics here if needed
        return loss

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
