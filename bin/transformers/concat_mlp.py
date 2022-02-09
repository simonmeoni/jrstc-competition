import torch
from torch import nn
from transformers import AutoTokenizer

from bin.transformers.hugginface_rm_dropout import rm_dropout
from bin.transformers.mlp import MLP


class ConcatMLP(nn.Module):
    def __init__(
        self, model, hidden_size, num_classes, tokenizer, max_length, remove_dropout
    ):
        super().__init__()
        self.model = rm_dropout(model, remove_dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.drop = nn.Dropout(p=0.2)
        self.fc = MLP(hidden_size * 4, num_classes)
        self.max_length = max_length

    def forward(self, text, device):
        tokens = self.tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
        ).to(device)
        out = torch.cat(
            self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )["hidden_states"][-4:],
            dim=2,
        )[:, 0, :]
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs
