from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.models.architectures.mlp import MLP
from src.utils.utils import rm_dropout


class SimpleMLP(nn.Module):
    def __init__(self, model, hidden_size, num_classes, tokenizer, max_length, remove_dropout):
        super().__init__()
        self.model = rm_dropout(model, remove_dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.drop = nn.Dropout(p=0.2)
        self.max_length = max_length
        self.fc = MLP(hidden_size, num_classes)

    def rm_dropout(model, remove_dropout):
        if remove_dropout:
            cfg = AutoConfig.from_pretrained(model)
            cfg.hidden_dropout_prob = 0
            cfg.attention_probs_dropout_prob = 0
            return AutoModel.from_pretrained(model, config=cfg)
        else:
            return AutoModel.from_pretrained(model)

    def forward(self, text, device):
        tokens = self.tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
        ).to(device)
        out = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=False,
        )
        out = out["last_hidden_state"][:, 0, :]
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs
