import gc

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def submission_loop(model, dataloader, device):
    dataset_size = 0
    test_predictions = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, batch in bar:
        outputs = model(batch["text"])
        test_predictions.append(outputs.view(-1).cpu().detach().numpy())

    test_predictions = np.concatenate(test_predictions)

    return test_predictions


def inference(model, checkpoints, test_loader, device, predictions):
    for i, checkpoint in enumerate(checkpoints):
        model.load_state_dict(torch.load(checkpoint)["state_dict"])
        print(f"Getting predictions for model {i + 1}")
        checkpoint_predictions = submission_loop(model, test_loader, device)
        predictions.append(checkpoint_predictions)
        gc.collect()
