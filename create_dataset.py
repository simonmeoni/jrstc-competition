import pandas as pd
from detoxify import Detoxify

from bin.file_utils import rm_and_new_folder
from bin.checkpoints.upload_to_kaggle import (
    kaggle_get_metadata,
    kaggle_new_dataset_version,
)

model = Detoxify("original")
data = pd.read_csv("data/jigsaw-toxic-severity-rating/validation_data.csv")
print(data.head())

data["concat_1"] = data["less_toxic"] + data["more_toxic"]
data["concat_2"] = data["more_toxic"] + data["less_toxic"]
all_pairs = list(set(list(data["concat_1"]) + list(data["concat_2"])))
data["concat_1"] = data["concat_1"].apply(lambda x: all_pairs.index(x))
data["concat_2"] = data["concat_2"].apply(lambda x: all_pairs.index(x))
data["concat_id"] = data.apply(
    (lambda x: "-".join(sorted([str(x["concat_1"]), str(x["concat_2"])]))), axis=1
)
data["concat_text"] = data["less_toxic"] + data["more_toxic"]
grouped_data = data.groupby(["concat_id"])
less_toxic = []
more_toxic = []
for key, item in grouped_data:
    group = grouped_data.get_group(key)
    most_voted_concat_text = group["concat_text"].value_counts().idxmax()
    most_voted = group.loc[group["concat_text"] == most_voted_concat_text].iloc[0]
    less_toxic.append(most_voted["less_toxic"])
    more_toxic.append(most_voted["more_toxic"])
print(data.head())

data = pd.DataFrame(data={"less_toxic": less_toxic, "more_toxic": more_toxic})
less_results = [model.predict(i) for i in list(data["less_toxic"])]
less_results = [
    (lambda x: {"less_" + el[0]: el[1] for el in x.items()})(i) for i in less_results
]
less_results = pd.DataFrame(data=less_results)

more_results = [model.predict(i) for i in list(data["more_toxic"])]
more_results = [
    (lambda x: {"more_" + el[0]: el[1] for el in x.items()})(i) for i in more_results
]
more_results = pd.DataFrame(data=more_results)

data = pd.concat([data, less_results, more_results], axis=1)
path = "data/jigsaw-classification-voting-cleaning"
rm_and_new_folder(path)
data.to_csv("data/jigsaw-classification-voting-cleaning/validation_data.csv")
kaggle_get_metadata(path=path, dataset_slug="jigsaw-classification-voting-cleaning")
kaggle_new_dataset_version(path=path)
