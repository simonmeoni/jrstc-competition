import json

import kaggle


def kaggle_new_dataset_version(path):
    api = kaggle.KaggleApi()
    api.authenticate()
    api.dataset_create_version(path, dir_mode="zip",
                               version_notes="new version",
                               convert_to_csv=False)


def kaggle_get_metadata(path, dataset_slug):
    api = kaggle.KaggleApi()
    api.authenticate()
    api.dataset_metadata(dataset=dataset_slug, path=path)
