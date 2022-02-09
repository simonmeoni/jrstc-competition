from torch import nn
from transformers import AutoTokenizer

from bin.transformers.hugginface_rm_dropout import rm_dropout
from bin.transformers.huggingface_freeze_layer import freeze_layer


class SimpleRegression(nn.Module):
    def __init__(
        self,
        model,
        hidden_size,
        num_classes,
        tokenizer,
        max_length,
        remove_dropout,
        freeze,
    ):
        super().__init__()
        self.model = rm_dropout(model, remove_dropout)
        self.model = freeze_layer(model, freeze)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.drop = nn.Dropout(p=0.2)
        self.max_length = max_length
        self.fc = nn.Linear(hidden_size, num_classes)

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
