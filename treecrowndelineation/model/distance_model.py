import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from ..modules import utils


architectures = [smp.Unet, smp.UnetPlusPlus, smp.Linknet, smp.FPN, smp.PSPNet, smp.PAN, smp.DeepLabV3,
                 smp.DeepLabV3Plus]
arch_names = [m.__name__.replace("Plus", "+") for m in architectures]
arch_dict = {name: m for name, m in zip(arch_names, architectures)}


class DistanceModel(pl.LightningModule):
    def __init__(self, in_channels: int, architecture: str = "Unet", backbone: str = "resnet18"):
        """ Distance transform model

        The model is the second part in the tree crown delineation model.

        Args:
            in_channels (int): Number of input channels
            architecture (str): One of 'Unet, Unet++, Linknet, FPN, PSPNet, PAN, DeepLabV3'
            backbone (str): One of the backbones supported by the [pytorch segmentation models package](https://github.com/qubvel/segmentation_models.pytorch)
        """
        super().__init__()
        arch = arch_dict[architecture]
        self.model = arch(encoder_name=backbone,
                          in_channels=in_channels,
                          classes=1,
                          encoder_depth=3,
                          decoder_channels=[64, 32, 16],
                          activation="sigmoid")
        # throw away unused weights
        self.model.encoder.layer3 = None
        self.model.encoder.layer4 = None
        utils.set_batchnorm_momentum(self.model, 0.99)

    def forward(self, img: torch.Tensor, mask_and_outline: torch.Tensor, from_logits: bool = False):
        """ Distance transform forward pass

        Args:
            img (torch.Tensor): Input image
            mask_and_outline (torch.Tensor): Tensor containing mask and outlines concatenated in channel dimension, \
                coming from the first sub-network.
            from_logits (bool): If set to true, sigmoid activation is applied to the mask_and_outline tensor.

        Returns:
            Model output of dimension N1HW
        """
        if from_logits:
            mask_and_outline = torch.sigmoid(mask_and_outline)

        x = torch.cat((img, mask_and_outline), dim=1)
        return self.model(x)

    def shared_step(self, batch):
        # x: raster
        # y: mask, outline, distance transform
        img, y = batch
        mask_and_outline = y[:, [0, 1]]
        distance_transform = y[:, [2]]
        y_pred = self(img, mask_and_outline)
        loss = torch.mean((y_pred - distance_transform) ** 2)
        return loss

    def training_step(self, batch, step):
        loss = self.shared_step(batch)
        self.log("train/loss_dist", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, step):
        loss = self.shared_step(batch)
        self.log("val/loss_dist", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1E-4, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 2)
        return [optimizer], [scheduler]
