from transformers import AutoModel


def freeze_layer(model, freeze_layer_flag):
    if freeze_layer_flag:
        model = AutoModel.from_pretrained(model)
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        return AutoModel.from_pretrained(model)
