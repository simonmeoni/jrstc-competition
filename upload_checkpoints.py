import argparse

from bin.file_utils import rm_and_new_folder
from bin.upload_to_kaggle import kaggle_get_metadata, kaggle_new_dataset_version
from bin.wandb_download_chekpoints import download


def main(checkpoints_path, kaggle_dataset, project, wandb_groups):
    rm_and_new_folder(checkpoints_path)
    for group in wandb_groups:
        download(group, project, checkpoints_path)
    kaggle_get_metadata(checkpoints_path, kaggle_dataset)
    kaggle_new_dataset_version(checkpoints_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="upload checkpoints to kaggle")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--kaggle_dataset", type=str)
    parser.add_argument("--checkpoints_path", type=str)
    parser.add_argument("--wandb_groups", nargs="+")

    args = parser.parse_args()
    main(
        checkpoints_path=args.checkpoints_path,
        kaggle_dataset=args.kaggle_dataset,
        project=args.wandb_project,
        wandb_groups=args.wandb_groups,
    )
