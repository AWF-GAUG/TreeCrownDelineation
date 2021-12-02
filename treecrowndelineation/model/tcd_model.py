import torch
import pytorch_lightning as pl
from .segmentation_model import SegmentationModel
from .distance_model import DistanceModel
from ..modules import metrics
from ..modules.losses import BinarySegmentationLoss


class TreeCrownDelineationModel(pl.LightningModule):
    def __init__(self, segmentation_model=None, distance_model=None, in_channels=None, lr: float = 1E-4):
        """Tree crown delineation model

        The model consists of two sub-netoworks (two U-Nets with ResNet backbone). The first network calculates a tree
        cover mask and the tree outlines, the second calculates the distance transform of the masks (distance to next
        background pixel). The first net receives the input image, the second one receives the input image and the output of network 1.

        Args:
            segmentation_model: pytorch model
            distance_model: pytorch model
            in_channels: Number of input channels / bands of the input image
            lr: learning rate
        """
        super().__init__()
        if in_channels is None and segmentation_model is not None and distance_model is not None:
            self.seg_model = segmentation_model
            self.dist_model = distance_model
        elif in_channels is not None and segmentation_model is None and distance_model is None:
            self.seg_model = SegmentationModel(in_channels=in_channels)
            self.dist_model = DistanceModel(in_channels=in_channels + 2)
        else:
            raise ValueError("Please provide *either* the base models or the number of input channels via "
                             "'in_channels'.")

        self.lr = lr

    def forward(self, img):
        mask_and_outline = torch.sigmoid(self.seg_model(img))
        dist = self.dist_model(img, mask_and_outline, from_logits=False)
        return torch.cat((mask_and_outline, dist), dim=1)

    def shared_step(self, batch):
        x, y = batch
        output = self(x)

        mask      = output[:, 0]
        outline   = output[:, 1]
        dist      = output[:, 2]

        mask_t    = y[:, 0]
        outline_t = y[:, 1]
        dist_t    = y[:, 2]

        iou_mask = metrics.iou(mask, mask_t)
        iou_outline = metrics.iou(outline, outline_t)

        loss_mask = BinarySegmentationLoss()(mask, mask_t)
        loss_outline = BinarySegmentationLoss()(outline, outline_t)
        loss_distance = torch.mean((dist - dist_t) ** 2)

        # lower mask loss results in unlearning the masks
        # lower distance loss results in artifacts in the distance transform
        loss = loss_mask + loss_outline + loss_distance

        return loss, loss_mask, loss_outline, loss_distance, iou_mask, iou_outline

    def training_step(self, batch, step):
        loss, loss_mask, loss_outline, loss_distance, iou_mask, iou_outline = self.shared_step(batch)
        self.log('train/loss'        , loss         , on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/loss_mask'   , loss_mask    , on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/loss_outline', loss_outline , on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/loss_dist'   , loss_distance, on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/iou_mask'    , iou_mask     , on_step = False, on_epoch = True, sync_dist = True)
        self.log('train/iou_outline' , iou_outline  , on_step = False, on_epoch = True, sync_dist = True)
        return loss

    def validation_step(self, batch, step):
        loss, loss_mask, loss_outline, loss_distance, iou_mask, iou_outline = self.shared_step(batch)
        self.log('val/loss'        , loss         , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_mask'   , loss_mask    , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_outline', loss_outline , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/loss_dist'   , loss_distance, on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou_mask'    , iou_mask     , on_step = False, on_epoch = True, sync_dist = True)
        self.log('val/iou_outline' , iou_outline  , on_step = False, on_epoch = True, sync_dist = True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 2)
        return [optimizer], [scheduler]

    @classmethod
    def from_checkpoint(cls, path: str,
                        architecture: str = "Unet", backbone: str = "resnet18", in_channels: int = 8):
        seg_model = SegmentationModel(architecture=architecture,
                                      backbone=backbone,
                                      in_channels=in_channels)
        dist_model = DistanceModel(in_channels=in_channels + 2)
        try:
            return cls.load_from_checkpoint(path, segmentation_model=seg_model, distance_model=dist_model)
        except NotImplementedError:
            return torch.jit.load(path)
