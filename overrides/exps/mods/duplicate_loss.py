import torch
import torch.nn as nn
from yolox.utils import bboxes_iou


class DuplicateLoss(nn.Module):
    def __init__(self, iou_threshold: float = 0.95, reduction: str ="none"):
        super(DuplicateLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.reduction = reduction

    def forward(self, bboxes_preds_per_image: torch.tensor):
        pair_wise_ious = bboxes_iou(bboxes_preds_per_image, bboxes_preds_per_image, False)

        pair_wise_ious = torch.triu(pair_wise_ious, diagonal=1)

        # only values above threshold matter
        loss = torch.clamp(pair_wise_ious - self.iou_threshold, min=0.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
