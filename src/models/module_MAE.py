from lightning import LightningModule
import torch
from hydra.utils import instantiate

class Module(LightningModule):
    def __init__(self, network, scheduler, optimizer):
        super().__init__()
        self.model = network.instance
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.model(x, 0.75, 0.75)

    def training_step(self, batch, batch_idx):
        loss, pred, mask = self.forward(batch)
        self.log(
            f"train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, pred, mask = self.forward(batch)
        self.log(
            f"val/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        pass

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss, pred, mask = self.forward(batch)
        self.log(
            f"test/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
