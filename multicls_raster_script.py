import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import subprocess
import glob
import os
import argparse
# from TreeCrownDelineation.scripts.rasterize_multiclass import filter_geometry, extent_to_poly
import rioxarray
import fiona
from multiprocessing import Pool
from shapely.geometry import shape, Polygon
import sqlite3
import shapely.wkb as wkb
import pandas as pd
import torch
import numpy as np
from torch import nn
from TreeCrownDelineation.treecrowndelineation.dataloading.in_memory_datamodule import (
    InMemoryDataModule,
)
from TreeCrownDelineation.treecrowndelineation.model.multiclass_model import (
    MultiClassTreeCrownDelineationModel,
)
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
import yaml


fiona.drvsupport.supported_drivers["SQLite"] = "rw"


class MetricsCallback(Callback):
    def __init__(self, data_module, device):
        self.data_module = data_module
        self.device = device

    def on_validation_epoch_end(self, trainer, pl_module):
        # Calculate IoU on the training dataset
        # train_iou, train_f1, train_precision, train_recall = self.calculate_metrics(
        train_iou, train_f1 = self.calculate_metrics(
            pl_module, self.data_module.train_dataloader(), "multiclass"
        )

        # Calculate IoU on the validation dataset
        # val_iou, val_f1, val_precision, val_recall = self.calculate_metrics(
        val_iou, val_f1 = self.calculate_metrics(
            pl_module, self.data_module.val_dataloader(), "multiclass"
        )

        # Log the IoU values
        trainer.logger.log_metrics(
            {
                "train/iou": train_iou, 
                "val/iou": val_iou, 
                "train/f1_score": train_f1, 
                "val/f1_score": val_f1,
                # "train/precision": train_precision,
                # "val/precision": val_precision,
                # "train/recall": train_recall,
                # "val/recall": val_recall,
            }, 
            step=trainer.global_step

        )

    def calculate_metrics(self, model, dataloader, mode="multiclass"):
        model.eval()
        iou_values, f1_values,precision_values, recall_values = [], [],[], []

        with torch.no_grad():
            for inputs, targets in dataloader:
                # Assuming inputs and targets are tensors, modify accordingly
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).squeeze(dim=1)

                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                # iou, f1, prec, recall = calc_metrics(
                iou, f1 = calc_metrics(
                    predictions.cpu().to(torch.uint8),
                    targets.cpu().to(torch.uint8),
                    mode=mode,
                )
                iou_values.append(iou)
                f1_values.append(f1)
                # precision_values.append(prec)
                # recall_values.append(recall)

        model.train()
        return torch.mean(torch.stack(iou_values)), torch.mean(torch.stack(f1_values))
    # , torch.mean(torch.stack(precision_values)), torch.mean(torch.stack(recall_values))


def calc_metrics(val_pred, val_gt, mode="multiclass", num_classes=14):
    tp, fp, fn, tn = smp.metrics.get_stats(
        output=val_pred - 1,
        target=val_gt - 1,
        mode=mode,
        ignore_index=-1,
        num_classes=num_classes,
    )

    return (
        smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"),
        smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro"),
        # smp.metrics.precision(tp, fp, fn, tn, reduction="micro"),
        # smp.metrics.recall(tp, fp, fn, tn, reduction="micro"),
    )


def execute_multiclass_training(train_config):
    LOGDIR = os.path.join(train_config["logdir"], train_config["experiment_name"])
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(train_config["model_save_path"], exist_ok=True)

    multicls_images = sorted(
        glob.glob(os.path.join(train_config["mcls_img_paths"], "**"))
    )
    multicls_masks = sorted(
        glob.glob(os.path.join(train_config["mcls_msk_paths"], "**"))
    )

    width = train_config["width"]
    train_augmentation = A.Compose(
        [
            A.RandomCrop(width, width, always_apply=True),
            A.RandomRotate90(),
            A.VerticalFlip(),
        ]
    )

    val_augmentation = A.RandomCrop(width, width, always_apply=True)

    data = InMemoryDataModule(
        rasters=multicls_images,
        targets=(multicls_masks,),
        width=width,
        batchsize=train_config["batchsize"],
        training_split=train_config["training_split"],
        train_augmentation=train_augmentation,
        val_augmentation=val_augmentation,
        concatenate_ndvi=train_config["concatenate_ndvi"],
        red=2,
        nir=3,
        #   dilate_second_target_band=2,
        rescale_ndvi=train_config["rescale_ndvi"],
    )

    model_name = "{}_{}_epochs={}_lr={}_width={}_bs={}".format(
        train_config["arch"],
        train_config["backbone"],
        train_config["max_epochs"],
        train_config["lr"],
        width,
        train_config["batchsize"],
    )

    model = MultiClassTreeCrownDelineationModel(
        in_channels=train_config["IN_CHS"],
        out_channels=train_config["OUT_CHS"],
        apply_softmax=train_config["apply_softmax"],
        lr=train_config["lr"],
        arch=train_config["arch"],
        backbone=train_config["backbone"],
    )

    # class ClassWeightsCallback(Callback):
    #     def __init__(self, data_module, device):
    #         self.data_module = data_module
    #         self.device = device

    #     def on_train_start(self, trainer, pl_module):
    #         pl_module.class_weights = self.data_module.get_class_weights().to(self.device)

    logger = TensorBoardLogger(
        LOGDIR,
        name=train_config["experiment_name"],
        version=model_name,
        # default_hp_metric=False,
    )

    cp = ModelCheckpoint(
        os.path.abspath(train_config["model_save_path"])
        + "/"
        + train_config["experiment_name"],
        model_name + "-{epoch}",
        monitor=train_config["monitor_metric"],
        save_last=train_config["save_last"],
        save_top_k=train_config["save_top_k"],
    )

    # class_weights_callback = ClassWeightsCallback(data, device=device)
    metrics_cb = MetricsCallback(data, device=train_config["device"])
    callbacks = [cp, metrics_cb, LearningRateMonitor()]

    trainer = Trainer(
        accelerator=train_config["accelerator"],
        devices=train_config["devices"],
        # strategy='ddp_notebook',
        num_nodes=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=train_config["max_epochs"],
    )

    trainer.fit(model, data)
    print("Training complete ✅")
    return data, model


def execute_multiclass_validation(data, model, save_cfg, save_preds=False):
    # After trainer.fit(model, data)

    # a) Prediction on Validation DataLoader
    val_dataloader = data.val_dataloader()

    # Put the model in evaluation mode
    model.eval()

    # Initialize lists to store predictions and ground truth masks
    val_predicted_labels = []
    val_ground_truth_labels = []

    # Iterate over the validation DataLoader
    with torch.no_grad():
        for inputs, masks in tqdm(val_dataloader):
            predictions = model(inputs)  # Forward pass
            predicted_labels = torch.argmax(
                predictions, dim=1
            )  # Apply argmax to get class labels
            val_predicted_labels.append(predicted_labels.cpu())
            val_ground_truth_labels.append(masks.cpu())

    # Convert the lists to NumPy arrays
    val_predicted_labels = np.concatenate(val_predicted_labels, axis=0)
    val_ground_truth_labels = np.concatenate(val_ground_truth_labels, axis=0).squeeze(axis=1)

    if save_preds:
        SAVE_DIR = f'./validation_data/{save_cfg["arch"]}_{save_cfg["backbone"]}'
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(f'{SAVE_DIR}/val_pred.npy', val_predicted_labels)
        np.save(f'{SAVE_DIR}/val_gt.npy', val_ground_truth_labels)

    print("Validation Done! ✅")

    # return val_predicted_labels, val_ground_truth_labels


from pprint import pprint

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training script for tree crown delineation model.')
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        data, model = execute_multiclass_training(config["training"])
        
        execute_multiclass_validation(data=data, model=model, save_cfg={
            'arch':config['training']['arch'],
            'backbone': config['training']['backbone']
        }, save_preds=True)
        
        
