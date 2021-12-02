import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from .metrics import iou, iou_with_logits


class BinarySegmentationLoss(_Loss):
    """Combines binary cross entropy loss with -log(iou).
    Works with probabilities, so after applying sigmoid activation."""

    def __init__(self, iou_weight=0.5, **kwargs):
        super().__init__()
        self.bceloss = nn.BCELoss(**kwargs)
        self.iou_weight = iou_weight

    def forward(self, y_pred, y_true):
        loss = (1 - self.iou_weight) * self.bceloss(y_pred, y_true)
        loss -= self.iou_weight * torch.log(iou(y_pred, y_true))
        return loss

class BinarySegmentationLossWithLogits(_Loss):
    """Combines binary cross entropy loss with -log(iou).
    Works with logits - don't apply sigmoid to your network output."""

    def __init__(self, iou_weight=0.5, **kwargs):
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss(**kwargs)
        self.iou_weight = iou_weight

    def forward(self, y_pred, y_true):
        loss = (1 - self.iou_weight) * self.bceloss(y_pred, y_true)
        loss -= self.iou_weight * torch.log(iou_with_logits(y_pred, y_true))
        return loss


# TODO: TEST ME
class WeightedBinarySegmentationLossWithLogits(_Loss):
    def __init__(self, iou_weight=0.5, reduction="mean", **kwargs):
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction="none", **kwargs)
        self.iou_weight = iou_weight
        self.reduction = reduction

    def forward(self, y_pred, y_true, weight_map):
        """
        Args:
            y_pred: NCHW
            y_true: NCHW
            weight_map: NCHW

        Returns:
            Loss
        """
        loss = (1 - self.iou_weight) * self.bceloss(y_pred, y_true)

        loss = weight_map * loss
        if self.reduction=="mean":
            loss = loss.mean()
        elif self.reduction=="sum":
            loss = loss.sum()
        else:
            raise NotImplementedError("Reduction '{}' not implemented.".format(self.reduction))

        loss -= self.iou_weight * torch.log(iou_with_logits(y_pred, y_true))
        return loss


# TODO: test me
class LossMultiWithLogits(_Loss):
    def __init__(self, iou_weight=0.5, num_classes=1, class_weights=None, reduction="sum"):
        super().__init__()
        self.cceloss = nn.CrossEntropyLoss(weight=weights, reduction=reduction)
        self.iou_weight = iou_weight
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        loss = (1 - self.iou_weight) * self.cceloss(y_pred, y_true)

        for cls in range(self.num_classes):
            output = y_pred[:, cls]
            target = y_true[:, cls]
            loss -= self.iou_weight * torch.log(iou_with_logits(output, target))
        return loss


class BinaryFocalLossWithLogits(nn.Module):
    """Binary focal loss with logits and reduction 'mean()'."""

    def __init__(self, gamma=2, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.bceloss = nn.BCEWithLogitsLoss(reduction="none", **kwargs)

    def forward(self, y_pred, y_true):
        p = torch.sigmoid(y_pred)
        return ((1 - p) ** self.gamma * self.bceloss(y_pred, y_true)).mean()


def direction_loss(y_pred, y_true, mask):
    eps = 1e-5
    # angle between y_true and y_pred, should ideally be zero
    angle = torch.arccos(torch.clip(torch.sum(y_pred * y_true, dim=1, keepdim=True), -1 + eps, 1 - eps))
    masked_angle = angle * mask
    total_error = torch.sum(masked_angle ** 2)
    return total_error / mask.sum()
