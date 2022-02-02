from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


class JTSRDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.more_toxic = df["more_toxic"].values
        self.less_toxic = df["less_toxic"].values
        self.less_toxic_target = None
        self.more_toxic_target = None
        if "more_toxic_target" in df.columns:
            self.more_toxic_target = df["more_toxic_target"].values
            self.less_toxic_target = df["less_toxic_target"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        more_toxic = self.more_toxic[index]
        less_toxic = self.less_toxic[index]
        target = 1
        if self.more_toxic_target is not None:
            more_toxic_target = self.more_toxic_target[index]
            less_toxic_target = self.less_toxic_target[index]
            return {
                "more_toxic": more_toxic,
                "more_toxic_target": more_toxic_target,
                "less_toxic": less_toxic,
                "less_toxic_target": less_toxic_target,
                "target": torch.tensor(target, dtype=torch.long),
            }
        return {
            "more_toxic": more_toxic,
            "less_toxic": less_toxic,
            "target": torch.tensor(target, dtype=torch.long),
        }


class DataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/jigsaw-toxic-severity-rating/validation_data.csv",
        val_data_dir: str = None,
        train_batch_size: int = 32,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_length: int = 128,
        k_fold: int = 5,
        current_fold: int = 0,
        tokenizer: str = "",
        train_data_size=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.full_dataset: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.full_dataset = (
            JTSRDataset(pd.read_csv(self.hparams.data_dir).dropna())
            if self.hparams.train_data_size is None
            else JTSRDataset(
                pd.read_csv(self.hparams.data_dir).dropna()[
                    : self.hparams.train_data_size
                ]
            )
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`,
        so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before
        trainer.fit()` or `trainer.test()`."""
        k_fold = KFold(n_splits=self.hparams.k_fold, shuffle=True)
        data_train_ids, data_val_ids = list(
            k_fold.split(self.full_dataset),
        )[self.hparams.current_fold]
        if self.hparams.k_fold == 0:
            self.data_train = JTSRDataset(pd.read_csv(self.hparams.data_dir).dropna())
            self.data_val = JTSRDataset(pd.read_csv(self.hparams.val_data_dir).dropna())
        elif self.hparams.val_data_dir:
            self.data_train = Subset(self.full_dataset, data_train_ids)
            self.data_val = JTSRDataset(pd.read_csv(self.hparams.val_data_dir).dropna())
        else:
            self.data_train = Subset(self.full_dataset, data_train_ids)
            self.data_val = Subset(self.full_dataset, data_val_ids)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        collate = torch.utils.data.dataloader.default_collate(batch)
        if "less_toxic_target" in collate.keys():
            return {
                "more_toxic": collate["more_toxic"],
                "more_toxic_target": collate["more_toxic_target"].float(),
                "less_toxic": collate["less_toxic"],
                "less_toxic_target": collate["less_toxic_target"].float(),
                "target": collate["target"],
            }
        return {
            "less_toxic": collate["less_toxic"],
            "more_toxic": collate["more_toxic"],
            "target": collate["target"],
        }
