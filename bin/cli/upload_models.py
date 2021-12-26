import os
import shutil

from transformers import AutoModel, AutoTokenizer

from bin.upload_to_kaggle import kaggle_get_metadata, kaggle_new_dataset_version

kaggle_dataset = "pretrainedmodels"
folder = "./models/"
pt_path = "./models/pt"
if os.path.exists(pt_path):
    shutil.rmtree(pt_path, ignore_errors=False)
os.makedirs(pt_path)


def download_huggingface_models(path, model_name):
    model = AutoModel.from_pretrained(model_name)
    model_path = model_name.replace("/", "_")
    model.save_pretrained(path + "/" + model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(path + "/" + model_path)


download_huggingface_models(pt_path, "unitary/unbiased-toxic-roberta")
download_huggingface_models(pt_path, "vinai/bertweet-base")
download_huggingface_models(pt_path, "unitary/toxic-bert")
download_huggingface_models(pt_path, "google/electra-base-discriminator")
download_huggingface_models(pt_path, "xlnet-base-cased")

kaggle_get_metadata(pt_path, kaggle_dataset)
kaggle_new_dataset_version(pt_path)
