import gc
import glob
import os.path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from torch import nn
from tqdm import tqdm

from bin.chunks import chunks
from bin.concat_regression import ConcatRegression
from bin.file_utils import rm_and_new_folder
from bin.seed_everything import seed_everything
from bin.upload_to_kaggle import kaggle_new_dataset_version, kaggle_get_metadata
from bin.wandb_download_chekpoints import download


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, device):
        return self.model(text, device)


def get_corpora(min_len=21, max_len=5000):
    jigsaw_bias = pd.read_csv(
        "data/jigsaw-unintended-bias-in-toxicity-classification/train.csv"
    )
    toxic_task = pd.read_csv("data/toxictask/task_a_distant.tsv", sep="\t")
    ccc = pd.read_csv(
        "https://raw.githubusercontent.com/ipavlopoulos/context_toxicity/master/data/CCC.csv"
    )
    unhealthy = pd.read_csv(
        "https://raw.githubusercontent.com/conversationai/unhealthy-conversations/main"
        "/unhealthy_full.csv"
    )
    jigsaw_toxic = pd.read_csv(
        "data/jigsaw-toxic-comment-classification-challenge/train.csv"
    )
    corpora = {
        "ccc": list(
            ccc[ccc["text"].str.len().between(min_len, max_len)]["text"].dropna()
        ),
        "unhealthy": list(
            unhealthy[unhealthy["comment"].str.len().between(min_len, max_len)][
                "comment"
            ].dropna()
        ),
        "jigsaw_bias": list(
            jigsaw_bias[
                jigsaw_bias["comment_text"].str.len().between(min_len, max_len)
            ][jigsaw_bias["target"] > 0]["comment_text"].dropna()
        ),
        "jigsaw_toxic": list(
            jigsaw_toxic[
                jigsaw_toxic["comment_text"].str.len().between(min_len, max_len)
            ]["comment_text"].dropna()
        ),
        "toxic_task_path": list(
            toxic_task[toxic_task["text"].str.len().between(min_len, max_len)]["text"].dropna()
        )[:300000],
    }
    jigsaw_severity = pd.read_csv(
        "data/jigsaw-classification-voting-cleaning/validation_data.csv"
    )
    return jigsaw_severity, [
        example for corpus in list(corpora.values()) for example in corpus
    ]


def load_models(config):
    checkpoints_path = (
        "models/checkpoints/concat_regression_unitary_unbiased-toxic-roberta_llrd"
    )
    if not os.path.isdir(
        "models/checkpoints/concat_regression_unitary_unbiased-toxic-roberta_llrd"
    ):
        checkpoints_path = download(
            "concat_regression/unitary/unbiased-toxic-roberta/llrd",
            "simonmeoni/jrstc-competition",
            "models/checkpoints",
        )

    concat_model = (
        Model(
            ConcatRegression(
                model="unitary/unbiased-toxic-roberta",
                hidden_size=config["hidden_size"],
                num_classes=config["num_classes"],
                tokenizer="unitary/unbiased-toxic-roberta",
                max_length=config["max_length"],
            )
        )
        .eval()
        .to(config["device"])
    )
    sentence_model = SentenceTransformer(
        "paraphrase-TinyBERT-L6-v2", device=config["device"]
    )
    return sentence_model, concat_model, checkpoints_path


def select_examples(comment, corpora_vectors, corpora, model, top_k=10):
    encoded_queries = model.encode(list(comment), convert_to_tensor=True)
    selected_sentences = []
    hits = util.semantic_search(
        query_embeddings=encoded_queries,
        corpus_embeddings=corpora_vectors,
        top_k=top_k,
    )
    for hit in hits:
        sentences = [corpora[h["corpus_id"]] for h in hit]
        selected_sentences.append(sentences)
    return selected_sentences


def predict(models_checkpoint, model, examples, device, batch_size=32):
    with torch.no_grad():
        predictions = []
        for checkpoint in glob.glob(models_checkpoint + "/*/*.ckpt"):
            model.load_state_dict(
                torch.load(checkpoint, map_location=device)["state_dict"]
            )
            checkpoint_preds = []
            for batch in chunks(examples, batch_size):
                checkpoint_preds.append(
                    model(batch, device).view(-1).detach().cpu().numpy()
                )
            gc.collect()
            predictions.append(np.concatenate(checkpoint_preds))
        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)
    return predictions


def encode_corpora(model, corpora):
    print("start corpora encoding ....")
    return model.encode(corpora, convert_to_tensor=True, show_progress_bar=True)


def upload_dataset(less_toxic_corpus, more_toxic_corpus):
    rm_and_new_folder("data/pseudo-jigsaw-severity")
    pd.DataFrame(
        data={"less_toxic": less_toxic_corpus, "more_toxic": more_toxic_corpus}
    ).to_csv("data/pseudo-jigsaw-severity/train.csv")
    kaggle_get_metadata("data/pseudo-jigsaw-severity", "pseudo-jigsaw-severity")
    kaggle_new_dataset_version("data/pseudo-jigsaw-severity")


def flat_excerpt(examples):
    return [example for batch_example in examples for example in batch_example]


def generate_dataset():
    config = {
        "seed": 42,
        "max_length": 128,
        "num_classes": 1,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "hidden_size": 768,
    }
    seed_everything(seed=config["seed"])
    val_corpus, train_corpora = get_corpora()
    sentence_model, ranking_model, checkpoints_path = load_models(config)
    less_toxic_corpus = []
    more_toxic_corpus = []
    train_corpora_vectors = encode_corpora(sentence_model, train_corpora)
    less_toxic_comment = val_corpus["less_toxic"]
    more_toxic_comment = val_corpus["more_toxic"]

    less_examples = flat_excerpt(select_examples(
        less_toxic_comment, train_corpora_vectors, train_corpora, sentence_model
    ))
    more_examples = flat_excerpt(select_examples(
        more_toxic_comment, train_corpora_vectors, train_corpora, sentence_model
    ))

    less_target = predict(
        checkpoints_path, ranking_model, less_examples, device=config["device"]
    )
    more_target = predict(
        checkpoints_path, ranking_model, more_examples, device=config["device"]
    )
    selected_examples_id = less_target < more_target
    less_toxic_corpus += less_toxic_corpus + [
        example
        for example, selected in zip(less_examples, selected_examples_id)
        if selected
    ]
    more_toxic_corpus += more_toxic_corpus + [
        example
        for example, selected in zip(more_examples, selected_examples_id)
        if selected
    ]
    upload_dataset(less_toxic_corpus, more_toxic_corpus)


if __name__ == "__main__":
    generate_dataset()
