import os
import shutil

from bin.transformers.download_huggingface_model import download_huggingface_model
from bin.checkpoints.upload_to_kaggle import (
    kaggle_get_metadata,
    kaggle_new_dataset_version,
)

kaggle_dataset = "pretrainedmodels"
folder = "./models/"
pt_path = "models/pt"
if os.path.exists(pt_path):
    shutil.rmtree(pt_path, ignore_errors=False)
os.makedirs(pt_path)


download_huggingface_model(pt_path, "roberta-base")
download_huggingface_model(pt_path, "microsoft/deberta-v3-base")
download_huggingface_model(pt_path, "unitary/unbiased-toxic-roberta")
download_huggingface_model(pt_path, "vinai/bertweet-base")
download_huggingface_model(pt_path, "unitary/toxic-bert")
download_huggingface_model(pt_path, "google/electra-base-discriminator")
download_huggingface_model(pt_path, "xlnet-base-cased")
download_huggingface_model(pt_path, "microsoft/deberta-v3-large")
download_huggingface_model(pt_path, "roberta-large")
download_huggingface_model(pt_path, "google/electra-large-discriminator")
download_huggingface_model(pt_path, "xlnet-large-cased")

kaggle_get_metadata(pt_path, kaggle_dataset)
kaggle_new_dataset_version(pt_path)
