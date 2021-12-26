import copy
import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from bin.concat_mlp import ConcatMLP
from bin.concat_regression import ConcatRegression
from bin.set_seed import set_seed
from bin.simple_regression import SimpleRegression
from bin.stacking_inference import inference, submission_loop
from bin.tfidf_mlp import TFIDFMLP
from bin.tfidf_regression import TFIDFRegression
from bin.tfidf_transformer import TFIDFTransformer


class JigsawDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.text = dataframe["text"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]

        return {"text": text}


MODELS = {
    {"arch": "simple_regression", "model": "roberta-base", "checkpoint": []},
    {"arch": "simple_regression", "model": "roberta-base", "checkpoint": []},
    {"arch": "simple_regression", "model": "roberta-base", "checkpoint": []},
    {"arch": "simple_regression", "model": "roberta-base", "checkpoint": []},
    {"arch": "simple_regression", "model": "roberta-base", "checkpoint": []},
}
CONFIG = {
    "seed": 42,
    "test_batch_size": 64,
    "max_length": 128,
    "num_classes": 1,
    "submission_path": "../input/jigsaw-toxic-severity-rating/comments_to_score.csv",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

set_seed(CONFIG["seed"])
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

predictions = []
df = pd.read_csv(CONFIG["submission_path"])


def initialize_model(arch, model_name):
    if arch == "concat_mlp":
        return ConcatMLP()
    elif arch == "concat_regression":
        return ConcatRegression()
    elif arch == "simple_regression":
        return SimpleRegression()
    elif arch == "tfidf_mlp":
        return TFIDFMLP()
    elif arch == "tfidf_regression":
        return TFIDFRegression()
    elif arch == "tfidf_transformer":
        return TFIDFTransformer()
    raise "not implemented yet !"


for model in MODELS:
    tokenizer = AutoTokenizer.from_pretrained(model["model"])
    model = initialize_model(arch=model["arch"], model_name=model["model"])
    model.to(CONFIG["device"])
    model.eval()
    print(df.head())

    test_dataset = JigsawDataset(df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["test_batch_size"],
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )
    inference(
        model=model,
        checkpoints=model["checkpoints"],
        test_loader=test_loader,
        device=CONFIG["device"],
        predictions=predictions,
    )

predictions = np.array(predictions)
predictions = np.mean(predictions, axis=0)
print(f"Total Predictions: {predictions.shape[0]}")
print(f"Total Unique Predictions: {np.unique(predictions).shape[0]}")
df["score"] = predictions
df.head()
df["score"] = df["score"].rank(method="first")
df.head()
df.drop("text", axis=1, inplace=True)
df.to_csv("submission.csv", index=False)
