import os
import glob
import numpy as np
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ..dataloading import datasets as ds
from ..modules.utils import dilate_img


class InMemoryDataModule(pl.LightningDataModule):
    def __init__(self,
                 rasters: str or list,
                 targets: tuple or list,
                 training_split: float = 0.7,
                 batchsize: int = 16,
                 width: int = 256,
                 train_augmentation=None,
                 val_augmentation=None,
                 use_last_target_as_mask=False,
                 concatenate_ndvi=False,
                 red=None,
                 nir=None,
                 divide_by=1,
                 dilate_second_target_band=False,
                 shuffle=True,
                 deterministic=True,
                 train_indices=None,
                 val_indices=None,
                 rescale_ndvi=True
                 ):
        """Pytorch lightning in memory data module

        For an explanation how data loading works look at README.md in dataloading source folder.

        Args:
            rasters (str or list): Can be a list of file paths or a path to a folder containing the training raster
                files in TIF format.
            targets: (tuple or list): This has to be either: a) A tuple of three folder-paths, e.g.
                (masks, outlines, distance_transforms) or b) a tuple / list of lists containg all the target files
            training_split (float): Value between 0 and 1 determining the training split. Default: 0.7
            batchsize (int): Batch size
            width (int): Width and height of the cropped images returned by the data loader.
            train_augmentation: Training augmentation from the albumentations package.
            val_augmentation: Validation augmentation from the albumentations package.
            use_last_target_as_mask (bool): This can be used to black out certain regions of the training data. To do this you
                have to load some no-data mask as last training target. Areas which you want to black out should be
                greater 1, areas where the mask is 0 are kept.
            concatenate_ndvi (bool): If set to true, the NDVI (normalized difference vegetation index) will be
                appended to the rasters.You have to get the red and near IR band indices.
            red (int): Index of the red band, starting from 0.
            nir (int): Index of the near IR band, starting from 0.
            divide_by (float): Constant value to divide the rasters by. Default: 1.
            dilate_second_target_band (int): The second target band (the tree outlines) can be dilated (widened) by a
                certain number of pixels.
            shuffle (bool): Whether or not to shuffle the data upon loading. This affects the partition into
                training and validation data. Default: True
            deterministic (bool): Enable or disables deterministic shuffling when loading the data. Set to True to
                always get the same data in the train and validation sets.
            train_indices (list): (Optional) List of indices specifying which images should be assigned to the training
            set.
            val_indices: (Optional) List of indices specifying which images should be assigned to the validation
            set.
            rescale_ndvi (bool): Whether to rescale the NDVI to the interval [0,1).
        """
        super().__init__()
        if type(rasters) in (list, tuple, np.ndarray):
            self.rasters = rasters
        else:
            self.rasters = np.sort(glob.glob(os.path.abspath(rasters) + "/*.tif"))

        if type(targets[0]) in (list, tuple, np.ndarray):
            self.targets = [np.sort(file_list) for file_list in targets]
        else:
            self.targets = [np.sort(glob.glob(os.path.abspath(file_list) + "/*.tif")) for file_list in targets]

        self.training_split = training_split
        self.batch_size = batchsize
        self.width = width
        self.train_augmentation = train_augmentation
        self.val_augmentation = val_augmentation
        self.use_last_target_as_mask = use_last_target_as_mask
        self.concatenate_ndvi = concatenate_ndvi
        self.red = red
        self.nir = nir
        self.dilate_second_target_band = dilate_second_target_band
        self.deterministic = deterministic
        self.shuffle = shuffle
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.divide_by = divide_by
        self.train_ds = None
        self.val_ds = None
        self.rescale_ndvi = rescale_ndvi

    def setup(self, unused=0):  # throws error if arg is removed
        if self.shuffle:
            for x in (self.rasters, *self.targets):
                if self.deterministic:
                    np.random.seed(1337)
                np.random.shuffle(x)  # in-place

        # split into training and validation set
        data = (self.rasters, *self.targets)

        # if traiing and validation indices are given, use them
        if self.train_indices is not None:
            training_data = [r[self.train_indices] for r in data]
        else:
            training_data = [r[:int(len(r) * self.training_split)] for r in data]

        if self.val_indices is not None:
            validation_data = [r[self.val_indices] for r in data]
        else:
            validation_data = [r[int(len(r) * self.training_split):] for r in data]

        # load the data into a custom dataset format
        self.train_ds = ds.InMemoryRSTorchDataset(training_data[0],
                                                  training_data[1:],
                                                  augmentation=self.train_augmentation,
                                                  cutout_size=(self.width, self.width),
                                                  dim_ordering="HWC",
                                                  divide_by=self.divide_by)

        if self.training_split < 1 or self.val_indices is not None:
            self.val_ds = ds.InMemoryRSTorchDataset(validation_data[0],
                                                    validation_data[1:],
                                                    augmentation=self.val_augmentation,
                                                    cutout_size=(self.width, self.width),
                                                    dim_ordering="HWC",
                                                    divide_by=self.divide_by)

        # attach the NDVI to the rasters
        if self.concatenate_ndvi and self.red is not None and self.nir is not None:
            # we rescale the NDVI to [0...1] to allow gamma augmentation to work right
            self.train_ds.concatenate_ndvi(red=self.red, nir=self.nir, rescale=self.rescale_ndvi)
            if self.training_split < 1 or self.val_indices is not None:
                self.val_ds.concatenate_ndvi(red=self.red, nir=self.nir, rescale=self.rescale_ndvi)

        # this can be used to black out certain regions, e.g. for those where no gt data is available
        if self.use_last_target_as_mask:
            for i, m in enumerate(self.train_ds.masks):
                mask = (1 - np.clip(m.data[..., -1] - m.data[..., 0], 0, 1))[..., None]
                self.train_ds.rasters[i] *= mask
                self.train_ds.masks[i] = self.train_ds.masks[i][:, :, :-1] * mask
            if self.training_split < 1 or self.val_indices is not None:
                for i, m in enumerate(self.val_ds.masks):
                    mask = (1 - np.clip(m.data[..., -1] - m.data[..., 0], 0, 1))[..., None]
                    self.val_ds.rasters[i] *= mask
                    self.val_ds.masks[i] = self.val_ds.masks[i][:, :, :-1] * mask

        # dilate the tree crown outlines to get a stronger training signal
        if self.dilate_second_target_band:
            for m in self.train_ds.masks:
                m[:, :, 1] = dilate_img(m[:, :, 1], self.dilate_second_target_band)
            if self.training_split < 1 or self.val_indices is not None:
                for m in self.val_ds.masks:
                    m[:, :, 1] = dilate_img(m[:, :, 1], self.dilate_second_target_band)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        if self.training_split == 1 and self.val_indices is None:
            return None
        else:
            return DataLoader(self.val_ds, batch_size=self.batch_size, drop_last=True, pin_memory=True)


class InMemoryMaskDataModule(InMemoryDataModule):
    def __init__(self,
                 rasters: str,
                 targets: tuple or list,
                 training_split: float = 0.7,
                 batchsize: int = 16,
                 width: int = 256,
                 use_last_target_as_mask=False,
                 concatenate_ndvi=False,
                 red=None,
                 nir=None,
                 divide_by=1,
                 dilate_second_target_band=False,
                 shuffle=True,
                 deterministic=True,
                 train_indices=None,
                 val_indices=None,
                 rescale_ndvi=True
                 ):
        """
        Please look at the documentation for `InMemoryDataModule`.
        """
        train_augmentation = A.Compose([  #A.RandomResizedCrop(width, width, scale=(0.25, 1.), always_apply=True),
            A.RandomCrop(width, width, always_apply=True),
                                        A.RandomRotate90(),
                                        A.VerticalFlip(),
            #A.RandomGamma(gamma_limit=(70, 130))
                                        ])
        val_augmentation = A.RandomCrop(width, width, always_apply=True)

        super().__init__(rasters, targets, training_split, batchsize, width,
                         train_augmentation, val_augmentation, use_last_target_as_mask,
                         concatenate_ndvi, red, nir, divide_by, dilate_second_target_band, shuffle, deterministic,
                         train_indices, val_indices, rescale_ndvi)
