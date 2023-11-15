#!/usr/bin/env python3
import os
import sys
import fiona.errors
import torch
import psutil
import traceback
import xarray as xr
import numpy as np
import pickle
import rioxarray
from torch.nn import DataParallel
from torch.nn import UpsamplingBilinear2d, Sequential
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import time
from fiona import crs
from treecrowndelineation.modules import utils
from treecrowndelineation.modules.indices import ndvi
from treecrowndelineation.modules.postprocessing import extract_polygons
from treecrowndelineation.model.inference_model import InferenceModel
from treecrowndelineation.model.averaging_model import AveragingModel


#%%
def get_parser():
    parser = ArgumentParser(description="This tool allows segmentation inference on many small images, e.g. GeoTiff.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_files",
                        dest="input_files",
                        type=str,
                        nargs='*',
                        help="Input raster file(s)",
                        required=True)
    parser.add_argument("-o", "--output_path",
                        dest="output_path",
                        type=str,
                        help="Output folder path",
                        required=True)
    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Path(s) to the model file(s) to load. "
                             "The model(s) must be loadable via torch.jit.load(), which means that "
                             "it has to be scripted or traced beforehand. For lightning models, use "
                             "to_torchscript(). If more than one model is given, the model outputs will be averaged.",
                        required=True,
                        nargs='*')
    parser.add_argument("-d", "--device",
                        dest="device",
                        type=str,
                        help="Device to run on. Default: 'cuda'",
                        default="cuda")
    parser.add_argument("-w", "--width",
                        dest="width",
                        type=int,
                        help="Desired network input window size during the prediction. "
                             "Must be divisible by 8.",
                        default=256)
    parser.add_argument("--red",
                        type=int,
                        help="Red channel index. Indexing starts with 0.",
                        default=0)
    parser.add_argument("--nir",
                        type=int,
                        help="Near infrared channel index. Indexing starts with 0.",
                        default=3)
    parser.add_argument("--ndvi",
                        dest="ndvi",
                        action="store_true",
                        help="If given, the NDVI index is calculated and appended to the data. "
                             "Red and near channel index can be specified.")
    parser.add_argument("-a", "--augmentation",
                        dest="augmentation",
                        action="store_true",
                        help="If given, the image is augmented by flipping and 90° rotation.")
    parser.add_argument("--min-dist",
                        dest="min_dist",
                        default=10,
                        type=int,
                        help="Minimum distance in pixels between local maxima during feature extraction.")
    parser.add_argument("--sigma",
                        dest="sigma",
                        default=2,
                        type=int,
                        help="Standard deviation of Gaussian filter during feature extraction.")
    parser.add_argument("-l", "--label-threshold",
                        dest="label_threshold",
                        default=0.5,
                        type=float,
                        help="Minimum height of local maxima during feature extraction.")
    parser.add_argument("-b", "--binary-threshold",
                        dest="binary_threshold",
                        default=0.1,
                        type=float,
                        help="Threshold value for the feature map; lower is background.")
    parser.add_argument("-s", "--simplify-dist",
                        dest="simplify",
                        default=0.3,
                        type=float,
                        help="Polygon simplification distance; vertices closer than this value are simplified.")
    parser.add_argument("--div", "--divide-by",
                        dest="divisor",
                        default=1.,
                        type=float,
                        help="Input data will be divided by this value,")
    parser.add_argument("--rescale-ndvi",
                        dest="rescale_ndvi",
                        default=False,
                        action="store_true",
                        help="If given, the NDVI is rescaled to 0...1.")
    parser.add_argument("--save-prediction",
                        dest="save_prediction",
                        type=str,
                        default=None,
                        help="Enables saving of intermediate neural network predictions. You can provide a base file "
                             "name / path to which the predictions will be saved as georeferenced tif.")
    parser.add_argument("--batchsize",
                        dest="batchsize",
                        type=int,
                        default=1,
                        help="Batchsize during inference.")
    parser.add_argument("--subsample",
                        dest="subsample",
                        default=False,
                        action="store_true",
                        help="If given, polygon extraction works at half the resolution of the model output. Use this for a speedup at the cost of accuracy.")
    parser.add_argument("--upsample",
                        dest="upsample",
                        default=1,
                        type=float,
                        help="The input to the neural network will be upsampled bilinearly by the given value.")
    parser.add_argument("--stride",
                        dest="stride",
                        default=None,
                        type=int,
                        help="Stride used when applying the network to the image.")
    parser.add_argument("--apply-sigmoid",
                        dest="apply_sigmoid",
                        action="store_true",
                        help="When this argument is given, the sigmoid activation will be applied to the mask and outline output.")
    return parser


def get_crs_wkt(array):
    return array.rio.crs.wkt


def get_transform(array):
    return array.rio.transform()


# n is chunksize
def split_into_chunks(list_, n):
    return [[list_[i * n:(i + 1) * n] for i in range((len(list_) + n - 1) // n )]]


if __name__ == '__main__':
    # args = get_parser().parse_args()
    args = get_parser().parse_args("-d cpu -i /data_hdd/bkg/clip_2021/1000.tif /data_hdd/bkg/clip_2023/10001.tif -o /data_hdd/bkg/vector/clip_2023/ -m /home/max/dr/models/bkg/Unet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=3_jitted.pt --div 255 --rescale-ndvi --ndvi --apply-sigmoid -a -w 512 --simplify 0.1".split())

    polygon_extraction_params = {"min_dist"          : args.min_dist,
                                 "mask_exp"          : 2,
                                 "outline_multiplier": 5,
                                 "dist_exp"          : 0.5,
                                 "sigma"             : args.sigma,
                                 "label_threshold"   : args.label_threshold,
                                 "binary_threshold"  : args.binary_threshold,
                                 "simplify"          : args.simplify
                                 }

    model_names = args.model

    print("Loading model")
    if len(model_names) == 1:
        model = torch.jit.load(args.model[0]).to(args.device)
    elif len(model_names) > 1:
        models = [torch.jit.load(m).to(args.device) for m in model_names]
        model = AveragingModel(models)
    else:
        print("Error during model loading.")
        sys.exit(1)

    print("Model loaded")

    stride = args.stride or args.width - 32 - args.width // 10

    if args.upsample != 1:
        model = Sequential(UpsamplingBilinear2d(scale_factor=args.upsample), model, UpsamplingBilinear2d(
                scale_factor=1. / args.upsample))

    if args.apply_sigmoid:
        model = InferenceModel(model)  # apply sigmoid to mask and outlines, but not to distance transform

    model = DataParallel(model)
    model.eval()

    inference_time = 0
    postprocessing_time = 0
    disk_loading_time = 0
    num_polygons = 0

    t0 = time()

    for idx, img in enumerate(args.input_files):
        filename = os.path.basename(img).split('.')[0]
        # array = xr.open_rasterio(img)
        array = rioxarray.open_rasterio(img)
        nbands, height, width = array.shape

        polygons = []

        print("Processing image {}/{}".format(idx+1, len(args.input_files)))
        t1 = time()
        chunk = array.load()  #.transpose('y', 'x', 'band')
        data = chunk.data
        if args.divisor != 1:
            data = data / args.divisor

        if args.ndvi:
            # data = np.concatenate((data, ndvi(data, red=args.red, nir=args.nir, axis=2)[...,None]), axis=2)
            ndvi_ = ndvi(data, red=args.red, nir=args.nir, axis=0)[None, ...]
            if args.rescale_ndvi:
                ndvi_ = (ndvi_ + 1) / 2
            data = np.concatenate((data, ndvi_), axis=0)

        t2 = time()
        disk_loading_time += t2 - t1
        result = utils.predict_on_array_cf(model,
                                           data,
                                           in_shape=(nbands + args.ndvi, args.width,
                                                     args.width),
                                           out_bands=3,
                                           drop_border=16,
                                           batchsize=args.batchsize,
                                           stride=stride,
                                           device=args.device,
                                           augmentation=args.augmentation,
                                           verbose=False)

        t3 = time()
        inference_time += t3 - t2

        if args.save_prediction is not None:
            utils.array_to_tif(result["prediction"].transpose(1, 2, 0),
                               os.path.abspath(args.save_prediction) + '/' + filename + "_pred.tif",
                               transform=utils.xarray_trafo_to_gdal_trafo(get_transform(chunk)),
                               crs=get_crs_wkt(array))

        t4 = time()

        if args.subsample:
            xres, xskew, xr, yskew, yres, yr = utils.get_xarray_trafo(chunk)
            xres *= 2
            yres *= 2
            trafo = (xres, xskew, xr, yskew, yres, yr)
            args.sigma /= 2
            args.min_dist /= 2

            polygons.extend(extract_polygons(*result["prediction"][:, ::2, ::2],
                                             transform=trafo,
                                             area_min=3,
                                             **polygon_extraction_params))

        else:
            trafo = get_transform(chunk)
            polygons.extend(extract_polygons(*result["prediction"],
                                             transform=trafo,
                                             area_min=3,
                                             **polygon_extraction_params))
        t5 = time()
        postprocessing_time += t5 - t4

        crs_ = get_crs_wkt(array)

        utils.save_polygons(polygons,
                            os.path.abspath(args.output_path) + '/' + filename + '.sqlite',
                            crs=crs_)

        num_polygons += len(polygons)

    print("Found {} polygons in total.".format(num_polygons))
    print("Total processing time: {}s".format(int(time() - t0)))
    print("Time loading from disk: {}s".format(int(disk_loading_time)))
    print("Inference time: {}s".format(int(inference_time)))
    print("Post-processing time: {}s".format(int(postprocessing_time)))
    print("Done processing {} images.".format(len(args.input_files)))
