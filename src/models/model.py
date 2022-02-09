from typing import Any, List

from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers import AdamW

from bin.transformers.huggingface_llrd import layerwise_learning_rate_decay
from bin.transformers.tfidf_mlp import TFIDFMLP
from bin.transformers.tfidf_regression import TFIDFRegression
from bin.transformers.tfidf_transformer import TFIDFTransformer
from bin.transformers.concat_mlp import ConcatMLP
from bin.transformers.concat_regression import ConcatRegression
from bin.transformers.simple_mlp import SimpleMLP
from bin.transformers.simple_regression import SimpleRegression
from src.models.losses.aux_loss import AuxLoss
from src.models.losses.margin_loss import MarginLoss
from src.utils.utils import Accuracy


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.mlp(x)


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
            architecture: str = "simple_regression",
            hidden_size: int = 768,
            margin: float = 0.5,
            num_classes: int = 1,
            scheduler=None,
            t_max: int = 700,
            eta_min: float = 0,
            loss: str = "margin_loss",
            eps: float = 1e-8,
            patience: int = 1,
            tokenizer: str = "",
            max_length: int = 128,
            tfidf_dir: str = "",
            remove_dropout: bool = False,
            freeze: bool = False,
            llrd: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        if self.hparams.architecture == "simple_regression":
            self.model = SimpleRegression(
                model=self.hparams.model,
                hidden_size=self.hparams.hidden_size,
                num_classes=self.hparams.num_classes,
                tokenizer=self.hparams.tokenizer,
                max_length=self.hparams.max_length,
                remove_dropout=self.hparams.remove_dropout,
                freeze=self.hparams.freeze,
            )
        elif self.hparams.architecture == "concat_regression":
            self.model = ConcatRegression(
                model=self.hparams.model,
                hidden_size=self.hparams.hidden_size,
                num_classes=self.hparams.num_classes,
                tokenizer=self.hparams.tokenizer,
                max_length=self.hparams.max_length,
                remove_dropout=self.hparams.remove_dropout,
                freeze=self.hparams.freeze,
            )
        elif self.hparams.architecture == "simple_mlp":
            self.model = SimpleMLP(
                model=self.hparams.model,
                hidden_size=self.hparams.hidden_size,
                num_classes=self.hparams.num_classes,
                tokenizer=self.hparams.tokenizer,
                max_length=self.hparams.max_length,
                remove_dropout=self.hparams.remove_dropout,
            )
        elif self.hparams.architecture == "concat_mlp":
            self.model = ConcatMLP(
                model=self.hparams.model,
                hidden_size=self.hparams.hidden_size,
                num_classes=self.hparams.num_classes,
                tokenizer=self.hparams.tokenizer,
                max_length=self.hparams.max_length,
                remove_dropout=self.hparams.remove_dropout,
            )
        elif self.hparams.architecture == "tfidf_regression":
            print("compute TF-IDF vector ...")
            self.model = TFIDFRegression(
                num_classes=self.hparams.num_classes,
                tf_idf_data=self.hparams.tfidf_dir,
            )
        elif self.hparams.architecture == "tfidf_mlp":
            print("compute TF-IDF vector ...")
            self.model = TFIDFMLP(
                num_classes=self.hparams.num_classes,
                tf_idf_data=self.hparams.tfidf_dir,
            )
            print("computing TF-IDF vector finished !")
        elif self.hparams.architecture == "tfidf_transformer":
            print("compute TF-IDF vector ...")
            self.model = TFIDFTransformer(
                num_classes=self.hparams.num_classes,
                tf_idf_data=self.hparams.tfidf_dir,
                model=self.hparams.model,
                hidden_size=self.hparams.hidden_size,
                tokenizer=self.hparams.tokenizer,
                max_length=self.hparams.max_length,
                remove_dropout=self.hparams.remove_dropout,
            )
            print("computing TF-IDF vector finished !")
        self.accuracy = Accuracy()

        # loss function
        if self.hparams.loss == "margin_loss":
            self.criterion = MarginLoss(margin=self.hparams.margin)
        elif self.hparams.loss == "aux_loss":
            self.criterion = AuxLoss(margin=self.hparams.margin)

    def forward(self, text):
        return self.model(text, self.device)

    def step(self, batch: Any):
        more_toxic_outputs = self.forward(batch["more_toxic"])
        less_toxic_outputs = self.forward(batch["less_toxic"])
        less_toxic_target = (
            batch["less_toxic_target"] if "less_toxic_target" in batch.keys() else None
        )
        more_toxic_target = (
            batch["more_toxic_target"] if "more_toxic_target" in batch.keys() else None
        )
        loss = self.criterion(
            less_toxic_outputs,
            more_toxic_outputs,
            less_toxic_target,
            more_toxic_target,
            batch["target"],
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
        more_toxic_outputs = self.forward(batch["more_toxic"])
        less_toxic_outputs = self.forward(batch["less_toxic"])
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
        elif self.hparams.scheduler == "ReduceLROnPlateau":
            return {
                "scheduler": ReduceLROnPlateau(optimizer, patience=self.hparams.patience),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.hparams.scheduler == "LinearScheduleWithWarmup":
            return {
                "scheduler": ReduceLROnPlateau(optimizer, patience=self.hparams.patience),
                "num_warmup_steps": 0,
                "num_training_steps": 0
            }
        return None

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        params = layerwise_learning_rate_decay(model=self.model,
                                               learning_rate=self.hparams.lr,
                                               weight_decay=self.hparams.weight_decay,
                                               layerwise_learning_rate_factor=0.7,
                                               layerwise_learning_rate_flag=self.hparams.llrd)
        optimizer = AdamW(
            params=params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
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
