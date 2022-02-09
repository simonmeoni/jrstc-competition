import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
import pandas as pd
from transformers import AutoTokenizer

from bin.transformers.hugginface_rm_dropout import rm_dropout


class TFIDFDense(nn.Module):
    def __init__(self, tf_idf_data):
        super().__init__()
        data = pd.read_csv(tf_idf_data)
        data = list(set(list(data["more_toxic"]) + list(data["less_toxic"])))
        self.tfidf = TfidfVectorizer(
            min_df=3, max_df=0.5, analyzer="char_wb", ngram_range=(3, 5)
        ).fit(data)
        self.mlp = nn.Sequential(
            nn.Linear(len(self.tfidf.get_feature_names()), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 100),
            nn.ReLU(),
        )

    def forward(self, text, device):
        vector = torch.Tensor(self.tfidf.transform(text).todense()).to(device)
        outputs = self.mlp(vector)
        return outputs


class TransformerDense(nn.Module):
    def __init__(self, model, hidden_size, tokenizer, max_length, remove_dropout):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = rm_dropout(model, remove_dropout)
        self.drop = nn.Dropout(p=0.2)
        self.max_length = max_length
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 100),
            nn.ReLU(),
        )

    def forward(self, text, device):
        tokens = self.tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
        ).to(device)
        transformer_outputs = torch.mean(
            torch.cat(
                self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    output_hidden_states=True,
                )["hidden_states"][-4:],
                dim=2,
            ),
            dim=1,
        )
        outputs = self.mlp(transformer_outputs)
        return outputs


class TFIDFTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        tf_idf_data,
        model,
        tokenizer,
        max_length,
        hidden_size,
        remove_dropout,
    ):
        super().__init__()
        self.tfidf_dense = TFIDFDense(tf_idf_data=tf_idf_data)
        self.transformer_dense = TransformerDense(
            model, hidden_size, tokenizer, max_length, remove_dropout
        )
        self.fc = nn.Linear(200, num_classes)

    def forward(self, text, device):
        tfidf_vector = self.tfidf_dense(text, device)
        transformers_vector = self.transformer_dense(text, device)
        concat_vector = torch.cat([tfidf_vector, transformers_vector], dim=1)
        return self.fc(concat_vector)
