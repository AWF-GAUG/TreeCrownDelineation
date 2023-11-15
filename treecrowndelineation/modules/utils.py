import os
import time
import osgeo.gdal as gdal
import osgeo.gdalnumeric as gdn
import numpy as np
import torch
import subprocess
import fiona
import operator
from sys import stdout
from uuid import uuid4
from skimage.morphology import dilation, square, disk
from shapely.geometry import Polygon, mapping, shape
from osgeo import osr


def load_model_weights(model, path):
    """Loads the models weights and sets the batch norm momentum to 0.9."""
    model.load_state_dict(torch.load(path))
    set_batchnorm_momentum(model, 0.9)
    return model


def gpu(x: torch.Tensor, device="cuda", dtype=torch.float32):
    if torch.cuda.is_available():
        return x.to(device=device, dtype=dtype)
    else:
        return x


def set_batchnorm_momentum(model, momentum):
    # print("Setting batchnorm momentum to {}.".format(momentum))
    if type(model) == torch.jit._script.RecursiveScriptModule:
        for m in model.modules():
            if "Batch" in m.original_name:
                m.momentum = momentum
    else:
        for m in model.modules():
            if "Batch" in str(m.__class__):
                m.momentum = momentum


def get_map_extent(gdal_raster):
    """Returns a dict of {xmin, xmax, ymin, ymax, xres, yres} of a given GDAL raster file.
    Returns None if no geo reference was found.
    Args:
        gdal_raster: File opened via gdal.Open().
    """
    xmin, xres, xskew, ymax, yskew, yres = gdal_raster.GetGeoTransform()
    xmax = xmin + (gdal_raster.RasterXSize * xres)
    ymin = ymax + (gdal_raster.RasterYSize * yres)
    # ret = ( (ymin, ymax), (xmin, xmax) )
    ret = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "xres": xres, "yres": yres}
    if 0. in (ymin, ymax, xmin, xmax): return None  # This is true if no real geodata is referenced.
    return ret


def gdal_trafo_to_xarray_trafo(gdal_trafo):
    xmin, xres, xskew, ymax, yskew, yres = gdal_trafo
    return (xres, xskew, xmin, yskew, yres, ymax)


def xarray_trafo_to_gdal_trafo(xarray_trafo):
    xres, xskew, xmin, yskew, yres, ymax = xarray_trafo
    return (xmin, xres, xskew, ymax, yskew, yres)


def get_xarray_extent(arr):
    """Returns
    xmin, xmax, ymin, ymax, xres, yres
    of an xarray. xres and yres can be negative.
    """
    xr = arr.coords["x"].data
    yr = arr.coords["y"].data
    xres, yres = (arr.transform[0], arr.transform[4])
    return min(xr), max(xr), min(yr), max(yr), xres, yres


def get_xarray_trafo(arr):
    """Returns
    xres, xskew, xmin, yskwe, yres, ymax
    """
    xr = arr.coords["x"].data
    yr = arr.coords["y"].data
    xres, yres = (arr.transform[0], arr.transform[4])
    xskew, yskew = (arr.transform[1], arr.transform[3])
    return xres, xskew, min(xr), yskew, yres, max(yr)


def extent_to_poly(xarr):
    """Returns the bounding box of an xarray as shapely polygon."""
    xmin, xmax, ymin, ymax, xres, yres = get_xarray_extent(xarr)
    return Polygon([(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)])


def load_filtered_polygons(file: str,
                           rasters: list,
                           minimum_area: float = 0,
                           maximum_area: float = 10 ** 6,
                           filter_dict: dict = {},
                           operators: list = [operator.eq]
                           ) -> list:
    """Loads those polygons from a given shapefile which fit into the extents of the given rasters.

    Polygons will be cropped to fit the given raster extent.

    Args:
        file (str): Shapefile path
        rasters (list): List of xarrays
        minimum_area (float): Minimum polygon area in map uniits (typically m²), measured after cropping to extent
        maximum_area (float): Maximum polygon area in map uniits (typically m²), measured after cropping to extent
        filter_dict (dict): Dictionary of key value pairs to filter polygons by, e.g. {"class": 1} - use this in
            conjunction with the operators arg to filter out polygons for which the operator returns true. E.g. pass
            'operators.eq' to test for equality; all polygons for which class is equal to 1 will be returned.
        operators (list): A list of built-in python comparison operators from the 'operator' package. See filter_dict.

    Returns:
        A list of lists containing polygons in the same order as the rasters.
    """

    def filter_polygons_by_property(p):
        for i, (k, v) in enumerate(filter_dict.items()):
            val = p["properties"][k]
            if operators[i](val, v):
                return True

    fiona.supported_drivers["SQLite"] = "rw"
    polygons = []
    with fiona.open(file) as src:
        for i, r in enumerate(rasters):
            xmin, xmax, ymin, ymax, xres, yres = get_xarray_extent(r)
            bbox = (xmin, ymin, xmax, ymax)
            crop = Polygon(((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)))

            tmp = []
            for p in src.filter(bbox=bbox):
                if len(filter_dict) > 0:
                    if not filter_polygons_by_property(p):
                        continue

                polygon = shape(p["geometry"])
                if not polygon.is_valid:
                    print("Skipping invalid polygon: {}".format(polygon))
                    continue

                intersection = crop.intersection(polygon)
                if minimum_area < intersection.area < maximum_area:
                    tmp.append(intersection)
            polygons.append(tmp)
    return polygons


def save_polygons(polygons: list, dest_fname: str, crs, driver: str = "SQLite", mode: str = "w"):
    """Save a list of polygons into a shapefile with given coordinate reference system.

    Args:
        polygons: List of shapely polygons to save
        dest_fname: Path to file
        crs: Coordinate reference system, e.g. from fiona.crs.from_epsg()
        driver: One of fiona's supported drivers e.g. 'ESRI Shapefile' or 'SQLite'
        mode: Either 'w' for write or 'a' for append. Not all drivers support both.
    """
    fiona.supported_drivers["SQLite"] = "rw"  # ensure we can actually write to a useful format
    schema = {"geometry"  : "Polygon",
              "properties": {"id": "int"}}

    records = [{"geometry": mapping(p), "properties": {"id": i}} for i, p in enumerate(polygons)]

    if os.path.exists(dest_fname):
        uuid = str(uuid4())[:4]
        head, suff = os.path.splitext(dest_fname)
        head = head + "_"
        dest_fname_new = uuid.join([head, suff])
        print("File {} already exists, saving as {}.".format(dest_fname, dest_fname_new))
        dest_fname = dest_fname_new

    with fiona.open(dest_fname, mode, crs=crs, driver=driver, schema=schema) as f:
        f.writerecords(records)


def read_img(input_file, dim_ordering="HWC", dtype='float32', band_mapping=None, return_extent=False):
    """Reads an image from disk and returns it as numpy array.

    Args:
        input_file: Path to the input file.
        dim_ordering: One of HWC or CHW, C=Channels, H=Height, W=Width
        dtype: Desired data type for loading, e.g. np.uint8, np.float32...
        band_mapping: Dictionary of which image band to load into which array band. E.g. {1:0, 3:1}
        return_extent: Whether or not to return the raster extent in the form (ymin, ymax, xmin, xmax). Defaults to False.

    Returns:
        Numpy array containing the image and optionally the extent.
    """
    if not os.path.isfile(input_file):
        raise RuntimeError("Input file does not exist. Given path: {}".format(input_file))

    ds = gdal.Open(input_file)
    extent = get_map_extent(ds)

    if band_mapping is None:
        num_bands = ds.RasterCount
        band_mapping = {i+1: i for i in range(num_bands)}
    elif isinstance(band_mapping, dict):
        num_bands = len(band_mapping)
    else:
        raise TypeError("band_mapping must be a dict, not {}.".format(type(band_mapping)))

    arr = np.empty((num_bands, ds.RasterYSize, ds.RasterXSize), dtype=dtype)

    for source_layer, target_layer in band_mapping.items():
        arr[target_layer] = gdn.BandReadAsArray(ds.GetRasterBand(source_layer))

    if dim_ordering == "HWC":
        arr = np.transpose(arr, (1, 2, 0))  # Reorders dimensions, so that channels are last
    elif dim_ordering == "CHW":
        pass
    else:
        raise ValueError("Dim ordering {} not supported. Choose one of 'HWC' or 'CHW'.".format(dim_ordering))

    if return_extent:
        return arr, extent
    else:
        return arr


def array_to_tif(array, dst_filename, num_bands='multi', save_background=True, src_raster: str = "", transform=None,
                 crs=None):
    """ Takes a numpy array and writes a tif. Uses deflate compression.

    Args:
        array: numpy array
        dst_filename (str): Destination file name/path
        num_bands (str): 'single' or 'multi'. If 'single' is chosen, everything is saved into one layer. The values
            in each layer of the input array are multiplied with the layer index and summed up. This is suitable for
            mutually exclusive categorical labels or single layer arrays. 'multi' is for normal images.
        save_background (bool): Whether or not to save the last layer, which is often the background class.
            Set to `True` for normal images.
        src_raster (str): Raster file used to determine the corner coords.
        transform: A geotransform in the gdal format
        crs: A coordinate reference system as proj4 string
    """
    if src_raster != "":
        src_raster = gdal.Open(src_raster)
        x_pixels = src_raster.RasterXSize
        y_pixels = src_raster.RasterYSize
    elif transform is not None and crs is not None:
        y_pixels, x_pixels = array.shape[:2]
    else:
        raise RuntimeError("Please provide either a source raster file or geotransform and coordinate reference "
                           "system.")

    bands = min( array.shape ) if array.ndim==3 else 1
    if not save_background and array.ndim==3: bands -= 1

    driver = gdal.GetDriverByName('GTiff')

    datatype = str(array.dtype)
    datatype_mapping = {'byte': gdal.GDT_Byte, 'uint8': gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16,
                        'uint32': gdal.GDT_UInt32, 'int8': gdal.GDT_Byte, 'int16': gdal.GDT_Int16,
                        'int32': gdal.GDT_Int32, 'float16': gdal.GDT_Float32, 'float32': gdal.GDT_Float32}
    options = ["COMPRESS=DEFLATE"]
    if datatype == "float16":
        options.append("NBITS=16")

    out = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1 if num_bands == 'single' else bands,
        datatype_mapping[datatype],
        options=options)

    if src_raster != "":
        out.SetGeoTransform(src_raster.GetGeoTransform())
        out.SetProjection(src_raster.GetProjection())
    else:
        out.SetGeoTransform(transform)
        srs = osr.SpatialReference()
        srs.ImportFromProj4(crs)
        out.SetProjection(srs.ExportToWkt())

    if array.ndim == 2:
        out.GetRasterBand(1).WriteArray(array)
        out.GetRasterBand(1).SetNoDataValue(0)
    else:
        if num_bands == 'single':
            singleband = np.zeros(array.shape[:2], dtype=array.dtype)
            for i in range(bands):
                singleband += (i+1)*array[:,:,i]
            out.GetRasterBand(1).WriteArray( singleband )
            out.GetRasterBand(1).SetNoDataValue(0)

        elif num_bands == 'multi':
            for i in range(bands):
                out.GetRasterBand(i+1).WriteArray( array[:,:,i] )
                out.GetRasterBand(i+1).SetNoDataValue(0)

    out.FlushCache()  # Write to disk.


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix is then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.

    Taken from & credit to:
        https://github.com/BloodAxe/pytorch-toolbelt/blob/f3acfca5da05cd7ccdd85e8d343d75fa40fb44d9/pytorch_toolbelt/inference/tiles.py#L16-L50

    Args:
        width: Tile width
        height: Tile height
    Returns:
        The weight mask as ndarray
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W


def predict_on_array(model,
                     arr,
                     in_shape,
                     out_bands,
                     stride=None,
                     drop_border=0,
                     batchsize=64,
                     dtype="float32",
                     device="cuda",
                     augmentation=False,
                     no_data=None,
                     verbose=False,
                     report_time=False):
    """
    Applies a pytorch segmentation model to an array in a strided manner.

    Call model.eval() before use!

    Args:
        model: pytorch model - make sure to call model.eval() before using this function!
        arr: HWC array for which the segmentation should be created
        stride: stride with which the model should be applied. Default: output size
        batchsize: number of images to process in parallel
        dtype: desired output type (default: float32)
        augmentation: whether to average over rotations and mirrorings of the image or not. triples computation time.
        no_data: a no-data vector. its length must match the number of layers in the input array.
        verbose: whether or not to display progress
        report_time: if true, returns (result, execution time)

    Returns:
        An array containing the segmentation.
    """
    t0 = None

    if augmentation:
        operations = (lambda x: x,
                      lambda x: np.rot90(x, 1),
                      # lambda x: np.rot90(x, 2),
                      # lambda x: np.rot90(x, 3),
                      # lambda x: np.flip(x,0),
                      lambda x: np.flip(x, 1))

        inverse = (lambda x: x,
                       lambda x: np.rot90(x, -1),
                       # lambda x: np.rot90(x, -2),
                       # lambda x: np.rot90(x, -3),
                       # lambda x: np.flip(x,0),
                       lambda x: np.flip(x,1))
    else:
        operations = (lambda x: x,)
        inverse = (lambda x: x,)

    assert in_shape[0] == in_shape[1], "Input shape must be equal in first two dims."
    out_shape = (in_shape[0] - 2 * drop_border, in_shape[1] - 2 * drop_border, out_bands)
    in_size = in_shape[0]
    out_size = out_shape[0]
    stride = stride or out_size
    pad = (in_size - out_size)//2
    assert pad % 2 == 0, "Model input and output shapes have to be divisible by 2."

    weight_mask = compute_pyramid_patch_weight_loss(out_size, out_size)

    original_size = arr.shape
    ymin = 0
    xmin = 0

    if no_data is not None:
        # assert arr.shape[-1]==len(no_data_vec), "Length of no_data_vec must match number of channels."
        # data_mask = np.all(arr[:,:,0].reshape( (-1,arr.shape[-1]) ) != no_data, axis=1).reshape(arr.shape[:2])
        nonzero = np.nonzero(arr[:,:,0]-no_data)
        ymin = np.min(nonzero[0])
        ymax = np.max(nonzero[0])
        xmin = np.min(nonzero[1])
        xmax = np.max(nonzero[1])
        img = arr[ymin:ymax, xmin:xmax]

    else:
        img = arr

    final_output = np.zeros(img.shape[:2]+(out_shape[-1],), dtype=dtype)

    op_cnt = 0
    for op, inv in zip(operations, inverse):
        img = op(img)
        img_shape = img.shape
        x_tiles = int(np.ceil(img.shape[1]/stride))
        y_tiles = int(np.ceil(img.shape[0]/stride))

        y_range = range(0, (y_tiles+1)*stride-out_size, stride)
        x_range = range(0, (x_tiles+1)*stride-out_size, stride)

        y_pad_after = y_range[-1]+in_size-img.shape[0]-pad
        x_pad_after = x_range[-1]+in_size-img.shape[1]-pad

        output = np.zeros( (img.shape[0]+y_pad_after-pad, img.shape[1]+x_pad_after-pad)+(out_shape[-1],), dtype=dtype)
        division_mask = np.zeros(output.shape[:2], dtype=dtype) + 1E-7
        img = np.pad(img, ((pad, y_pad_after), (pad, x_pad_after), (0, 0)), mode='reflect')

        patches = len(y_range)*len(x_range)

        def patch_generator():
            for y in y_range:
                for x in x_range:
                    yield img[y:y+in_size, x:x+in_size]

        patch_gen = patch_generator()

        y = 0
        x = 0
        patch_idx = 0
        batchsize_ = batchsize

        t0 = time.time()

        while patch_idx<patches:
            batchsize_ = min(batchsize_, patches, patches - patch_idx)
            patch_idx += batchsize_
            if verbose: stdout.write("\r%.2f%%" % (100 * (patch_idx + op_cnt * patches) / (len(operations) * patches)))

            batch = np.zeros((batchsize_,) + in_shape, dtype=dtype)

            for j in range(batchsize_):
                batch[j] = next(patch_gen)

            with torch.no_grad():
                prediction = model(
                    torch.from_numpy(batch.transpose((0, 3, 1, 2))).to(device=device, dtype=torch.float32))
                # prediction = torch.sigmoid(prediction)
                prediction = prediction.detach().cpu().numpy()
                prediction = prediction.transpose((0, 2, 3, 1))
            if drop_border > 0:
                prediction = prediction[:, drop_border:-drop_border, drop_border:-drop_border, :]

            for j in range(batchsize_):
                output[y:y + out_size, x:x + out_size] += prediction[j] * weight_mask[..., None]
                division_mask[y:y + out_size, x:x + out_size] += weight_mask
                x += stride
                if x + out_size > output.shape[1]:
                    x = 0
                    y += stride

        output = output / division_mask[..., None]
        output = inv(output[:img_shape[0], :img_shape[1]])
        final_output += output
        img = arr[ymin:ymax, xmin:xmax] if no_data is not None else arr
        op_cnt += 1
        if verbose: stdout.write("\rAugmentation step %d/%d done.\n" % (op_cnt, len(operations)))

    if verbose: stdout.flush()

    final_output = final_output/len(operations)

    if no_data is not None:
        final_output = np.pad(final_output, ((ymin, original_size[0]-ymax),(xmin, original_size[1]-xmax),(0,0)), mode='constant', constant_values=0)

    if report_time:
        return final_output, time.time() - t0

    else:
        return final_output


def predict_on_array_cf(model,
                        arr,
                        in_shape,
                        out_bands,
                        stride=None,
                        drop_border=0,
                        batchsize=64,
                        dtype="float32",
                        device="cuda",
                        augmentation=False,
                        no_data=None,
                        verbose=False,
                        aggregate_metric=False):
    """
    Applies a pytorch segmentation model to an array in a strided manner.

    Channels first version.

    Call model.eval() before use!

    Args:
        model: pytorch model - make sure to call model.eval() before using this function!
        arr: CHW array for which the segmentation should be created
        stride: stride with which the model should be applied. Default: output size
        batchsize: number of images to process in parallel
        dtype: desired output type (default: float32)
        augmentation: whether to average over rotations and mirrorings of the image or not. triples computation time.
        no_data: a no-data vector. its length must match the number of layers in the input array.
        verbose: whether or not to display progress
        aggregate_metric: This is for development purposes or for active learning. In case the model returns
            (prediction, some_metric), some_metric will be summed up for all predictions necessary to process the
            input image. The model can then e.g. be an ensemble model, returning the result and the variance.

    Returns:
        A dict containing result, time, nodata_region and time
    """
    t0 = time.time()
    metric = 0

    if augmentation:
        operations = (lambda x: x,
                      lambda x: np.rot90(x, 1, axes=(1, 2)),
                      # lambda x: np.rot90(x, 2),
                      # lambda x: np.rot90(x, 3),
                      # lambda x: np.flip(x,0),
                      lambda x: np.flip(x, 1))

        inverse = (lambda x: x,
                   lambda x: np.rot90(x, -1, axes=(1, 2)),
                   # lambda x: np.rot90(x, -2),
                   # lambda x: np.rot90(x, -3),
                   # lambda x: np.flip(x,0),
                   lambda x: np.flip(x, 1))
    else:
        operations = (lambda x: x,)
        inverse = (lambda x: x,)

    assert in_shape[1] == in_shape[2], "Input shape must be equal in last two dims."
    out_shape = (out_bands, in_shape[1] - 2 * drop_border, in_shape[2] - 2 * drop_border)
    in_size = in_shape[1]
    out_size = out_shape[1]
    stride = stride or out_size
    pad = (in_size - out_size) // 2
    assert pad % 2 == 0, "Model input and output shapes have to be divisible by 2."

    original_size = arr.shape
    ymin = 0
    xmin = 0
    ymax = arr.shape[0]
    xmax = arr.shape[1]

    if no_data is not None:
        # assert arr.shape[-1]==len(no_data_vec), "Length of no_data_vec must match number of channels."
        # data_mask = np.all(arr[:,:,0].reshape( (-1,arr.shape[-1]) ) != no_data, axis=1).reshape(arr.shape[:2])
        nonzero = np.nonzero(arr[0, :, :] - no_data)
        if len(nonzero[0]) == 0:
            return {"prediction": None,
                    "time": time.time() - t0,
                    "nodata_region": (0, 0, 0, 0),
                    "metric": metric}

        ymin = np.min(nonzero[0])
        ymax = np.max(nonzero[0])
        xmin = np.min(nonzero[1])
        xmax = np.max(nonzero[1])
        img = arr[:, ymin:ymax, xmin:xmax]

    else:
        img = arr

    weight_mask = compute_pyramid_patch_weight_loss(out_size, out_size)
    final_output = np.zeros((out_bands,) + img.shape[1:], dtype=dtype)

    op_cnt = 0
    for op, inv in zip(operations, inverse):
        img = op(img)
        img_shape = img.shape
        x_tiles = int(np.ceil(img.shape[2] / stride))
        y_tiles = int(np.ceil(img.shape[1] / stride))

        y_range = range(0, (y_tiles + 1) * stride - out_size, stride)
        x_range = range(0, (x_tiles + 1) * stride - out_size, stride)

        y_pad_after = y_range[-1] + in_size - img.shape[1] - pad
        x_pad_after = x_range[-1] + in_size - img.shape[2] - pad

        output = np.zeros((out_bands,) + (img.shape[1] + y_pad_after - pad, img.shape[2] + x_pad_after - pad),
                          dtype=dtype)
        division_mask = np.zeros(output.shape[1:], dtype=dtype) + 1E-7
        img = np.pad(img, ((0, 0), (pad, y_pad_after), (pad, x_pad_after)), mode='reflect')

        patches = len(y_range) * len(x_range)

        def patch_generator():
            for y in y_range:
                for x in x_range:
                    yield img[:, y:y + in_size, x:x + in_size]

        patch_gen = patch_generator()

        y = 0
        x = 0
        patch_idx = 0
        batchsize_ = batchsize

        t0 = time.time()

        while patch_idx < patches:
            batchsize_ = min(batchsize_, patches, patches - patch_idx)
            patch_idx += batchsize_
            if verbose: stdout.write("\r%.2f%%" % (100 * (patch_idx + op_cnt * patches) / (len(operations) * patches)))

            batch = np.zeros((batchsize_,) + in_shape, dtype=dtype)

            for j in range(batchsize_):
                batch[j] = next(patch_gen)

            with torch.no_grad():
                prediction = model(torch.from_numpy(batch).to(device=device, dtype=torch.float32))
                if aggregate_metric:
                    metric += prediction[1].cpu().numpy()
                    prediction = prediction[0]

                prediction = prediction.detach().cpu().numpy()
            if drop_border > 0:
                prediction = prediction[:, :, drop_border:-drop_border, drop_border:-drop_border]

            for j in range(batchsize_):
                output[:, y:y + out_size, x:x + out_size] += prediction[j] * weight_mask[None, ...]
                division_mask[y:y + out_size, x:x + out_size] += weight_mask
                x += stride
                if x + out_size > output.shape[2]:
                    x = 0
                    y += stride

        output = output / division_mask[None, ...]
        output = inv(output[:, :img_shape[1], :img_shape[2]])
        final_output += output
        img = arr[:, ymin:ymax, xmin:xmax] if no_data is not None else arr
        op_cnt += 1
        if verbose: stdout.write("\rAugmentation step %d/%d done.\n" % (op_cnt, len(operations)))

    if verbose: stdout.flush()

    final_output = final_output / len(operations)

    if no_data is not None:
        final_output = np.pad(final_output,
                              ((0, 0), (ymin, original_size[1] - ymax), (xmin, original_size[2] - xmax)),
                              mode='constant',
                              constant_values=0)

    return {"prediction": final_output,
            "time": time.time() - t0,
            "nodata_region": (ymin, ymax, xmin, xmax),
            "metric": metric}


def calc_band_stats(fpath : str):
    means = []
    stddevs = []
    p = subprocess.Popen(["gdalinfo -approx_stats {}".format(fpath)], shell=True, stdout=subprocess.PIPE)
    ret = p.stdout.readlines()
    p.wait()
    for s in ret:
        s = s.decode("UTF-8")
        if s.startswith("  Minimum"):
            m = float(s.split(",")[2].split("=")[-1])
            std = float(s.split(",")[3].split("=")[-1])
            means.append(m)
            stddevs.append(std)
    return {"mean" : np.array(means), "stddev" : np.array(stddevs)}


def dilate_img(img, size=10, shape="square"):
    if shape == "square":
        selem = square(size)
    elif shape == "disk":
        selem = disk(size)
    else:
        ValueError("Unknown shape {}, choose 'square' or 'disk'.".format(shape))
    return dilation(img, selem)


def write_info_file(path, **kwargs):
    file = open(path,'w')
    for key, value in kwargs.items():
        file.write( "{}: {}\n".format(key, value) )
    file.close()

