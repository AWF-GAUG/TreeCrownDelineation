import ctypes
import torch
import shapely
import xarray as xr
import numpy as np
import multiprocessing as mp
from torch.utils.data import IterableDataset, get_worker_info
from modules.indices import ndvi
from modules.utils import read_img


class InMemoryRSDataset:
    """In memory remote sensing dataset for image segmentation."""

    def __init__(self, raster_files: list, mask_files: list, dim_ordering="HWC", dtype="float32", divide_by=1,
                 overwrite_nan_with_zeros=True):
        """Creates a dataset containing images and masks which resides in memory.

        Args:
            raster_files: List of file paths to source rasters. File names must be of the form '.../the_name_i.tif' where i is some index
            mask_files: A tuple containing lists of file paths to different sorts of 'masks',
                e.g. mask, outline, distance transform.
                The mask and raster file names must have the same index ending.
            dim_ordering: One of HWC or CHW; how rasters and masks are stored in memory. The albumentations library
                needs HWC, so this is the default. CHW support could be bugged.
            dtype: Data type for storing rasters and masks
        """
        # initial sanity checks
        assert len(raster_files) > 0, "List of given rasters is empty."
        for i, m in enumerate(mask_files):
            if len(m) == 0:
                raise RuntimeError("Mask list {} is empty.".format(i))
            if len(m) != len(raster_files):
                raise RuntimeError("The length of the given lists must be equal.")
            for j, r in enumerate(raster_files):
                raster_file_index = r.split('.')[-2].split('_')[-1]
                mask_file_index = m[j].split('.')[-2].split('_')[-1]
                if raster_file_index != mask_file_index:
                    raise RuntimeError("The raster and mask lists must be sorted equally.")

        self.raster_files = raster_files
        self.mask_files = mask_files
        # self.functions = functions
        self.divide_by = divide_by

        self.rasters = []
        self.masks = []
        self.num_bands = 0
        self.num_mask_bands = 0
        self.dim_ordering = dim_ordering
        self.native_bands = 0
        self.dtype = dtype
        self.overwrite_nan = overwrite_nan_with_zeros
        self.weights = None  # can be used for proportional sampling of unevenly sized tiles
        if dim_ordering == "CHW":
            self.chax = 0  # channel axis of imported arrays
        elif dim_ordering == "HWC":
            self.chax = 2
        else:
            raise ValueError("Dim ordering {} not supported. Choose one of 'CHW' or 'HWC'.".format(dim_ordering))
        self.lateral_ax = np.array((1,2)) if self.chax==0 else np.array((0,1))

        self.load_data()

    def load_raster(self, file: str, used_bands: list = None):
        """Loads a raster from disk.

        Args:
            file (str): file to load
            used_bands (list): bands to use, indexing starts from 0, default 'None' loads all bands
        """
        arr = xr.open_rasterio(file).load().astype(self.dtype)  # eagerly load the image from disk via_load
        arr.close()  # dont know if needed, but to be sure...


        num_bands = arr.shape[0]
        if len(self.rasters) == 0: self.num_bands = num_bands
        if self.num_bands != num_bands:
            raise ValueError(
                "Number of raster layers ({}) does not match previous raster layer count ({}).".format(num_bands,
                                                                                                       self.num_bands))

        if used_bands is not None:
            arr = arr[used_bands]

        if self.dim_ordering == "HWC":
            arr = arr.transpose('y', 'x', 'band')

        arr.data /= self.divide_by  # dividing the array directly loses information on transformation etc?!?? wtf?

        self.rasters.append(arr)
        self.native_bands = np.arange(self.num_bands)
        self.weights = self.get_raster_weights()

    def load_mask(self, file: str):
        """Loads a mask from disk."""
        arr = xr.open_rasterio(file).load().astype(self.dtype)  # eagerly load the image from disk via load
        arr.close()  # dont know if needed, but to be sure...

        num_bands = arr.shape[0]
        if len(self.masks) == 0: self.num_mask_bands = num_bands
        # if self.num_mask_bands != num_bands:
        #     raise ValueError(
        #         "Number of mask layers ({}) does not match previous mask layer count({}).".format(num_bands,
        #                                                                                           self.num_mask_bands))

        if self.dim_ordering == "HWC":
            arr = arr.transpose('y', 'x', 'band')

        if self.overwrite_nan:
            nanmask = np.isnan(arr.data)
            arr.data[nanmask] = 0

        return arr
        # self.masks.append(arr)

    def load_data(self):
        for r in self.raster_files:
            self.load_raster(r)

        for files in zip(*self.mask_files):
            masks = [self.load_mask(f) for f in files]
            # "override" ensures that small differences in geotransorm are neglected
            mask = xr.concat(masks, dim="band", join="override")
            self.masks.append(mask)

    def apply_to_rasters(self, f):
        """Applies function f to all rasters."""
        for i, r in enumerate(self.rasters):
            self.rasters[i].data[:] = f(r.data).astype(self.dtype)

    def apply_to_masks(self, f):
        """Applies function f to all rasters."""
        for i, m in enumerate(self.masks):
            self.masks[i].data[:] = f(m.data).astype(self.dtype)

    def concatenate_ndvi(self, red=3, nir=4, rescale=False):
        for i, r in enumerate(self.rasters):
            # res = ndvi_xarray(r, red, nir).expand_dims(dim="band", axis=self.chax)
            if rescale:
                res = (ndvi(r, red, nir, axis=self.chax).expand_dims(dim="band", axis=self.chax) + 1) / 2
            else:
                res = ndvi(r, red, nir, axis=self.chax).expand_dims(dim="band", axis=self.chax)
            res = res.assign_coords(band=[self.num_bands + 1])  # this line took 3 hours
            self.rasters[i] = xr.concat((r, res), dim="band")
        self.num_bands += 1

    def get_raster_weights(self):
        """Returns size of img i divided by total dataset size."""
        weights = [np.prod(np.array(r.shape)[self.lateral_ax]) for r in self.rasters]
        weights /= np.sum(weights)
        return weights

    def calculate_dataset_mean_stddev(self, estimate=True):
        """Calculates the mean and standard deviation of the whole dataset."""
        mean = np.zeros(self.num_bands)
        stddev = np.zeros(self.num_bands)
        weights = self.get_raster_weights()
        for i, r in enumerate(self.rasters):
            if estimate:
                if self.dim_ordering == "HWC":
                    mean += np.mean(r[::4, ::4, :], axis=tuple(self.lateral_ax)) * weights[i]
                    stddev += np.std(r[::4, ::4, :], axis=tuple(self.lateral_ax)) * weights[i]
                else:
                    mean += np.mean(r[:, ::4, ::4], axis=tuple(self.lateral_ax)) * weights[i]
                    stddev += np.std(r[:, ::4, ::4], axis=tuple(self.lateral_ax)) * weights[i]
            else:
                mean  += np.mean(r, axis=tuple(self.lateral_ax)) * weights[i]
                stddev += np.std(r, axis=tuple(self.lateral_ax)) * weights[i]
        return mean.data, stddev.data

    def normalize(self):
        """Normalizes each raster by the mean and standard deviation of the whole dataset."""
        mean, stddev = self.calculate_dataset_mean_stddev()
        f = lambda x: (x-mean)/(stddev+1E-5)
        self.apply_to_rasters(f)

    def normalize_percentile(self, low=2, high=99):
        """Normalizes each raster by a weighted percentile of the whole raster."""
        weights = self.get_raster_weights()
        minp = [np.percentile(r.data, low) * weights[i] for i, r in enumerate(self.rasters)]
        maxp = [np.percentile(r.data, high) * weights[i] for i, r in enumerate(self.rasters)]
        minp = np.sum(minp)
        maxp = np.sum(maxp)
        f = lambda x: np.clip((x - minp) / (maxp - minp + 1E-5), 0, 1)
        self.apply_to_rasters(f)


class InMemoryRSTorchDataset(InMemoryRSDataset, IterableDataset):
    def __init__(self, raster_files: list, mask_files: list, augmentation, cutout_size, dim_ordering="HWC",
                 dtype="float32", divide_by=1):
        super().__init__(raster_files, mask_files, dim_ordering, dtype, divide_by)
        self.augment = augmentation  # or (lambda x, y: (x, y))
        self.cutout_size = cutout_size

    # these two methods are needed for pytorch dataloaders to work
    def __len__(self):
        # sum of product of all raster sizes
        total_pixels = np.sum([np.prod(np.array(r.shape)[self.lateral_ax]) for r in self.rasters])
        # product of the shape of cutout done by the transformation
        # uses albumentation augmentation API
        cutout_pixels = np.prod(np.array(self.cutout_size)[self.lateral_ax])
        return int(total_pixels / cutout_pixels)

    def __iter__(self):
        i = 0
        while i < len(self):
            idx = np.random.choice(np.arange(len(self.rasters)), p=self.weights)
            augmented = self.augment(image=self.rasters[idx].data,
                                     mask=self.masks[idx].data)  # giving the data only should speed things up!
            image = augmented["image"].transpose((2, 0, 1))
            mask = augmented["mask"].transpose((2, 0, 1))
            i += 1
            yield image, mask


class RSDataset:
    """Remote sensing dataset. Simple container class."""

    def __init__(self, rasters: list, masks: tuple, functions: tuple):
        # some inital sanity checks for equal length and sorting, O(n^2)!
        for i, m in enumerate(masks):
            if len(m) != len(rasters):
                raise RuntimeError("The length of the given lists must be equal.")
            for j, r in enumerate(rasters):
                raster_file_index = r.split('.')[-2].split('_')[-1]
                mask_file_index = m[j].split('.')[-2].split('_')[-1]
                if raster_file_index != mask_file_index:
                    raise RuntimeError("The raster and mask lists must be sorted equally.")

        self.rasters = rasters
        self.masks = masks
        self.functions = functions
        self.i = 0
        self.current_raster = None
        self.current_mask = None

    def load_next(self):
        """Load the next set of raster and mask into memory and apply optional functions to the masks.

        Cycles through the rasters infinitely.

        Args:
            functions: A tuple of functions. One for each kind of mask to load.
                       Use 'None' as a placeholder for identity.
        """
        if self.functions is not None:
            assert len(self.functions) == len(self.masks), "Number of functions must match " \
                                                           "the number of different kinds of masks"

        raster = self.rasters[self.i]
        masks = [mask[self.i] for mask in self.masks]
        raster = read_img(raster, dim_ordering="HWC", dtype=np.uint8)
        masks = [read_img(mask, dim_ordering="HWC", dtype=np.float32) for mask in masks]
        if self.functions is not None:
            for i, f in enumerate(self.functions):
                if f is not None:
                    masks[i] = [f(m) for m in masks[i]]
        mask = np.concatenate(masks, axis=2)
        self.current_raster = raster
        self.current_mask = mask
        self.i = (self.i + 1) % len(self.rasters)


class RSTorchDataset(RSDataset, IterableDataset):
    def __init__(self, rasters, masks, augmentation, cutout_size,
                 functions=None, sampling_ratio=1, num_workers=0, dtype="float32"):
        super().__init__(rasters, masks, functions)
        self.augmentation = augmentation
        self.cutout_size = cutout_size
        self.sampling_ratio = sampling_ratio
        self.length = self._calculate_len()
        self._current_len = 0
        self.total_idx = 0
        self.local_idx = 0
        self.dtype = dtype
        self.num_workers = num_workers
        self.raster_index_state = None
        if self.num_workers > 0:
            shared_array = mp.Array(ctypes.c_int, self.num_workers)
            self.raster_index_state = np.ctypeslib.as_array(shared_array.get_obj())
            self.raster_index_state[:] = 0

    def _calculate_len(self):
        # go through all rasters, check shape, sum up, divide by cutout size
        total_size = 0
        for r in self.rasters:
            arr = xr.open_rasterio(r)
            total_size += np.prod(np.array(arr.shape)[1:])
            arr.close()
        return total_size // np.prod(self.cutout_size)

    def _calculate_current_len(self):
        self._current_len = int(np.prod(np.array(self.current_raster.shape)[:2]) \
                                // np.prod(self.cutout_size)
                                * self.sampling_ratio)

    def __len__(self):
        return self.length

    def __next__(self):
        if self.local_idx >= self._current_len:
            # worker_info = get_worker_info()
            # id_ = worker_info.id if worker_info is not None else ""
            # print("worker {} loading image {}".format(id_, self.rasters[self.i]))
            self.load_next()
            self.local_idx = 0
            self._calculate_current_len()

        augmented = self.augmentation(image=self.current_raster, mask=self.current_mask)
        image = augmented["image"].transpose((2, 0, 1))
        mask = augmented["mask"].transpose((2, 0, 1))
        self.local_idx += 1
        self.total_idx += 1
        return image.astype(self.dtype), mask.astype(self.dtype)

    def __iter__(self):
        if self.num_workers > 0:
            self.i = self.raster_index_state[get_worker_info().id]

        while self.total_idx <= self.length:
            image, mask = next(self)
            yield image, mask

        if self.num_workers > 0:
            self.raster_index_state[get_worker_info().id] = self.i

        self.total_idx = 0  # important so that the iterator can restart

    @staticmethod
    def worker_init_fn(id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        rasters = dataset.rasters
        masks = dataset.masks
        l = len(rasters) // worker_info.num_workers
        local_rasters = rasters[id * l:(id + 1) * l]
        local_masks = tuple([m[id * l:(id + 1) * l] for m in masks])
        dataset.rasters = local_rasters
        dataset.masks = local_masks
        dataset.length //= worker_info.num_workers
