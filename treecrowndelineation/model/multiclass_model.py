import torch
import pytorch_lightning as pl
from .segmentation_model import SegmentationModel
from ..modules import metrics
from ..modules.losses import BinarySegmentationLossWithLogits, MultiClassCrossEntropyLossWithLogits


class MultiClassTreeCrownDelineationModel(pl.LightningModule):
    def __init__(self, segmentation_model=None, in_channels=None,out_channels=None, lr=1E-4, apply_softmax=False, arch='Unet', backbone='resnet18'):
        """Multi class Tree crown delineation model

        Args:
            segmentation_model: pytorch model
            in_channels: Number of input channels / bands of the input image
            lr: learning rate
        """
        super().__init__()
        if in_channels is None and segmentation_model is not None:
            self.seg_model = segmentation_model
        elif in_channels is not None and segmentation_model is None:
            self.seg_model = SegmentationModel(in_channels=in_channels, out_channels=out_channels, architecture=arch, backbone=backbone)
        else:
            raise ValueError("Please provide *either* the base models or the number of input channels via "
                             "'in_channels'.")

        self.lr = lr
        self.apply_softmax = apply_softmax
        self.class_weights = None
        self.num_classes = out_channels

    def forward(self, img):
        mask_and_outline = self.seg_model(img)
        # dist = dist * mask_and_outline[:, [0]]
        if self.apply_softmax:
            return torch.softmax(mask_and_outline,dim=1)
        else:
            return mask_and_outline

    def shared_step(self, batch):
        x, y = batch
        output = self(x)
        # print(output.shape, y.shape)

        mask      = output
        mask_t    = y[:, 0].to(torch.long)

        # iou_mask = metrics.iou(torch.argmax(torch.softmax(mask,dim=1), dim=1), mask_t)

        # loss_mask = BinarySegmentationLossWithLogits(reduction="mean")(mask, mask_t)
        loss_mask = MultiClassCrossEntropyLossWithLogits(num_classes=self.num_classes, class_weights=self.class_weights)(mask, mask_t)
        # loss_outline = BinarySegmentationLossWithLogits(reduction="mean")(outline, outline_t)
        # loss_distance = torch.mean((dist - dist_t) ** 2)

        # lower mask loss results in unlearning the masks
        # lower distance loss results in artifacts in the distance transform

        # return loss_mask, iou_mask
        return loss_mask

    def training_step(self, batch, step):
        # loss_mask, iou_mask = self.shared_step(batch)
        loss_mask = self.shared_step(batch)
        self.log('train/loss_mask'   , loss_mask    , on_step = False, on_epoch = True, sync_dist = True)
        # self.log('train/iou_mask'    , iou_mask     , on_step = False, on_epoch = True, sync_dist = True)
        return loss_mask

    def validation_step(self, batch, step):
        # loss_mask, iou_mask = self.shared_step(batch)
        loss_mask = self.shared_step(batch)
        self.log('val/loss_mask'   , loss_mask    , on_step = False, on_epoch = True, sync_dist = True)
        # self.log('val/iou_mask'    , iou_mask     , on_step = False, on_epoch = True, sync_dist = True)
        return loss_mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 2)
        return [optimizer], [scheduler]

    @classmethod
    def from_checkpoint(cls, path: str, architecture: str = "Unet", backbone: str = "resnet18", in_channels: int = 8):
        seg_model = SegmentationModel(architecture=architecture, backbone=backbone,in_channels=in_channels)
        try:
            return cls.load_from_checkpoint(path, segmentation_model=seg_model)
        except NotImplementedError:
            return torch.jit.load(path)
