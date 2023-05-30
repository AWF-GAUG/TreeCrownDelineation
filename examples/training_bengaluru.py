# This code has not been tested against the latest version of the code

import os
import torch
import albumentations as A
import pytorch_lightning as pl

from treecrowndelineation.model.segmentation_model import SegmentationModel
from treecrowndelineation.model.distance_model import DistanceModel
from treecrowndelineation.model.tcd_model import TreeCrownDelineationModel
from treecrowndelineation.dataloading.in_memory_datamodule import InMemoryDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == '__main__':
    ###################################
    #      file paths and settings    #
    ###################################
    rasters = "/data/bangalore/training_data/treecrown_delineation/tiles"
    masks = "/data/bangalore/training_data/treecrown_delineation/masks"
    outlines = "/data/bangalore/training_data/treecrown_delineation/outlines"
    dist = "/data/bangalore/training_data/treecrown_delineation/dist_trafo"

    rasters_pretrain = "/data/bangalore/training_data/treecover_segmentation/tiles_north"
    masks_pretrain = "/data/bangalore/training_data/treecover_segmentation/masks_north"
    outlines_pretrain = "/data/bangalore/training_data/treecover_segmentation/outlines_north"

    logdir = "/home/max/dr/log"
    model_save_path = "/home/max/dr/models"
    experiment_name = "bengaluru"

    arch = "Unet-resnet18"
    width = 256
    batchsize = 16
    in_channels = 8
    gpus = 2
    backend = "dp"
    max_epochs = 30 + 60 - 1
    max_pretrain_epochs = 200
    lr = 3E-4

    training_split = 0.8

    model_name = "{}_epochs={}_lr={}_width={}_bs={}_ts=1".format(arch,
                                                                 max_epochs,
                                                                 lr,
                                                                 width,
                                                                 batchsize)

    #%%
    ###################################
    #           pre-training          #
    ###################################
    # pre-train on 330 ha of tree cover masks @ 0.3m resolution

    logger = TensorBoardLogger(logdir,
                               name=experiment_name + "_pre-training",
                               version=model_name,
                               default_hp_metric=False)

    cp = ModelCheckpoint(os.path.abspath(model_save_path) + "/" + experiment_name,
                         model_name + "_pre-training" + "-{epoch}",
                         monitor="train/loss",
                         save_last=True,
                         save_top_k=2)

    callbacks = [cp, LearningRateMonitor()]

    train_augmentation = A.Compose([A.RandomCrop(width, width, always_apply=True),
                                    A.RandomRotate90(),
                                    A.VerticalFlip()
                                    ])
    val_augmentation = A.RandomCrop(width, width, always_apply=True)

    data_pretraining = InMemoryDataModule(rasters_pretrain,
                                          (masks_pretrain, outlines_pretrain),
                                          width=width,
                                          batchsize=batchsize,
                                          training_split=training_split,
                                          train_augmentation=train_augmentation,
                                          val_augmentation=val_augmentation,
                                          concatenate_ndvi=True,
                                          red=3,
                                          nir=4,
                                          dilate_second_target_band=2,
                                          rescale_ndvi=True)

    mask_model = SegmentationModel(in_channels=in_channels, lr=lr)

    trainer = Trainer(gpus=gpus,
                      distributed_backend=backend,
                      logger=logger,
                      callbacks=callbacks,
                      # checkpoint_callback=False,  # set this to avoid logging into the working directory
                      max_epochs=max_pretrain_epochs)
    trainer.fit(mask_model, data_pretraining)

    #%%
    ###################################
    #             training            #
    ###################################
    logger = TensorBoardLogger(logdir,
                               name=experiment_name,
                               version=model_name,
                               default_hp_metric=False)

    cp = ModelCheckpoint(os.path.abspath(model_save_path) + "/" + experiment_name,
                         model_name + "-{epoch}",
                         monitor="train/loss",
                         save_last=True,
                         save_top_k=2)

    # swa = pl.callbacks.StochasticWeightAveraging(70, annealing_epochs=0)

    callbacks = [cp, LearningRateMonitor()]

    data = InMemoryDataModule(rasters,
                              (masks, outlines, dist),
                              width=width,
                              batchsize=batchsize,
                              training_split=1,
                              train_augmentation=train_augmentation,
                              val_augmentation=val_augmentation,
                              concatenate_ndvi=True,
                              red=3,
                              nir=4,
                              dilate_second_target_band=2,
                              rescale_ndvi=True)

    # instantiate the rest of the model and reuse the pre-trained segmentation part
    dist_model = DistanceModel(in_channels=in_channels + 2)
    full_model = TreeCrownDelineationModel(mask_model, dist_model, lr=lr)
    #%%
    trainer = Trainer(gpus=gpus,
                      distributed_backend=backend,
                      logger=logger,
                      callbacks=callbacks,
                      # checkpoint_callback=False,  # set this to avoid logging into the working directory
                      max_epochs=max_epochs)
    trainer.fit(full_model, data)
#%%
    full_model.to("cpu")
    t = torch.rand(1, in_channels, width, width, dtype=torch.float32)
    full_model.to_torchscript(
        os.path.abspath(model_save_path) + "/" + experiment_name + '/' + model_name + "_jitted.pt",
        method="trace",
        example_inputs=t)
