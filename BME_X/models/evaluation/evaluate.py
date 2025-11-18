from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.DenseUNet3d import DenseUNet3d
from models.evaluation.dice_score import dice_score


def evaluate(
    model: DenseUNet3d,
    device: torch.device,
    dataloader: DataLoader,
    dim: Optional[int] = 1,
) -> float:
    """
    Evaluates the model by dice score on given data

    :param model:       model to evaluate
    :param device:      device to evaluate on, usually cpu or gpu
    :param dataloader:  DataLoader object which contains the images
    :param dim:         dimension to evaluate dice score over
    :return:            average dice score
    """
    dice_scores = []

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            volume, segmentation = data
            volume = volume.to(device, dtype=torch.float)

            # Tumors are livers too
            if dim == 1:
                segmentation = torch.clamp(segmentation, 0, 1)

            segmentation = segmentation.to(device, dtype=torch.long)

            output = model(volume)
            output = F.one_hot(torch.argmax(output, dim=1), num_classes=3).permute(
                4, 0, 1, 2, 3
            )
            segmentation = F.one_hot(segmentation.squeeze(1), num_classes=3).permute(
                4, 0, 1, 2, 3
            )

            dice_scores.append(dice_score(output, segmentation))

    return sum(dice_scores) / len(dice_scores)
