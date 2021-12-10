from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AdamW, AutoModel

from src.utils.utils import Accuracy


class Model(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        model: str = "",
        margin: float = 0.5,
        num_classes: int = 1,
        scheduler=None,
        t_max: int = 0,
        eta_min: float = 0,
        mse_loss: bool = False,
        marge_ranking_loss: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model = AutoModel.from_pretrained(self.hparams.model)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, self.hparams.num_classes)
        self.accuracy = Accuracy()

        # loss function
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=self.hparams.margin)
        self.mse_loss = nn.MSELoss()

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs

    def step(self, batch: Any):
        more_toxic_outputs = self.forward(
            batch["more_toxic"]["input_ids"], batch["more_toxic"]["attention_mask"]
        )
        less_toxic_outputs = self.forward(
            batch["less_toxic"]["input_ids"], batch["less_toxic"]["attention_mask"]
        )
        loss = None
        if self.hparams.marge_ranking_loss:
            loss = self.margin_ranking_loss(
                more_toxic_outputs, less_toxic_outputs, batch["target"]
            )
        if self.hparams.mse_loss:
            less_mse_loss = self.mse_loss(more_toxic_outputs, batch["more_toxic_target"])
            more_mse_loss = self.mse_loss(less_toxic_outputs, batch["less_toxic_target"])
            loss = (
                loss * 0.5 + (less_mse_loss + more_mse_loss) * 0.5
                if loss is not None
                else less_mse_loss * 0.5 + more_mse_loss * 0.5
            )
        return less_toxic_outputs, more_toxic_outputs, loss

    def training_step(self, batch: Any, batch_idx: int):
        less_toxic, more_toxic, loss = self.step(batch)
        # log train metrics
        self.accuracy(less_toxic, more_toxic)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.accuracy, on_step=True, on_epoch=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        less_toxic, more_toxic, loss = self.step(batch)
        self.accuracy(less_toxic, more_toxic)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        more_toxic_outputs = self.forward(
            batch["more_toxic"]["input_ids"], batch["more_toxic"]["attention_mask"]
        )
        less_toxic_outputs = self.forward(
            batch["less_toxic"]["input_ids"], batch["less_toxic"]["attention_mask"]
        )
        self.accuracy(less_toxic_outputs, more_toxic_outputs)
        self.log("test/acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        pass

    def fetch_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            return {
                "scheduler": CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.t_max,
                    eta_min=self.hparams.eta_min,
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            return None

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = self.fetch_scheduler(optimizer)
        if scheduler is None:
            return {
                "optimizer": optimizer,
            }
        return {
            "lr_scheduler": scheduler,
            "optimizer": optimizer,
        }
