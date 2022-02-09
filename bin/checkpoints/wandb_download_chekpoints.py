import wandb
import os


def download(group, project, checkpoints_path):
    normalized_group = group.replace("/", "_")
    api = wandb.apis.public.Api()
    runs = api.runs(path=project)
    group_path = checkpoints_path + "/" + normalized_group
    os.makedirs(group_path)
    for run in runs:
        if run.group == group:
            run = api.run(project + "/" + run.id)
            for artifact in run.logged_artifacts():
                if artifact.type == "checkpoints":
                    api.artifact(name=project + "/" + artifact.name, type="checkpoints")
                    artifact.download(root=group_path + "/" + run.name)
    return checkpoints_path + "/" + normalized_group
