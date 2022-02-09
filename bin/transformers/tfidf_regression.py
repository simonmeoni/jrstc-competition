import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
import pandas as pd


class TFIDFRegression(nn.Module):
    def __init__(self, num_classes, tf_idf_data):
        super().__init__()
        data = pd.read_csv(tf_idf_data)
        data = list(set(list(data["more_toxic"]) + list(data["less_toxic"])))
        self.tfidf = TfidfVectorizer(
            min_df=3, max_df=0.5, analyzer="char_wb", ngram_range=(3, 5)
        ).fit(data)
        self.fc = nn.Linear(len(self.tfidf.get_feature_names()), num_classes)

    def forward(self, text, device):
        vector = torch.Tensor(self.tfidf.transform(text).todense()).to(device)
        outputs = self.fc(vector)
        return outputs
