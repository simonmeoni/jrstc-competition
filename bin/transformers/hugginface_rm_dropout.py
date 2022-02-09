from transformers import AutoConfig, AutoModel


def rm_dropout(model, remove_dropout):
    if remove_dropout:
        cfg = AutoConfig.from_pretrained(model)
        cfg.hidden_dropout_prob = 0
        cfg.attention_probs_dropout_prob = 0
        return AutoModel.from_pretrained(model, config=cfg)
    else:
        return AutoModel.from_pretrained(model)
