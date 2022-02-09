from transformers import AutoModel, AutoTokenizer


def download_huggingface_model(path, model_name):
    model = AutoModel.from_pretrained(model_name)
    model_path = model_name.replace("/", "_")
    model.save_pretrained(path + "/" + model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(path + "/" + model_path)
