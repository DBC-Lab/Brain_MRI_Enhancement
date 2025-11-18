from typing import Optional

import torch


def dice_score(img1: torch.Tensor, img2: torch.Tensor, dim: Optional[int] = 1) -> float:
    """
    Compute dice score between two images in a given dimension

    :param img1:    first tesnor
    :param img2:    second tensor
    :param dim:     dimension to compute dice score from
                    In this project, 0 is background, 1 is liver, and 2 is tumor.
    """
    diff = img1[dim] * img2[dim]
    intersect = diff.sum()

    mag1 = img1[dim].sum()
    mag2 = img2[dim].sum()

    dice = 2 * intersect / (mag1 + mag2)

    return dice.item()
