#!/usr/bin/env python3
import sys
import fiona.errors
import torch
import psutil
import traceback
import xarray as xr
import numpy as np
import pickle
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


def get_parser():
    parser = ArgumentParser(description="This tool allows segmentation inference on large images, e.g. GeoTiff.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_file",
                        dest="input_file",
                        type=str,
                        # nargs='*',
                        help="Input raster file",
                        required=True)
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        help="Output file path and name, e.g. ~/output/result.sqlite",
                        required=True)
    parser.add_argument("-m", "--model",
                        dest="model",
                        type=str,
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
                        default=16,
                        help="Batchsize during inference.")
    parser.add_argument("--tilesize",
                        dest="tilesize",
                        type=int,
                        default=0,
                        help="Tile size for reading chunks from disk. "
                             "Default is 0, which triggers an estimation based on available RAM.")
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
    return parser


def get_crs(array):
    crs_ = array.attrs["crs"]
    if "epsg" in crs_:
        crs_ = crs.from_epsg(crs_.split(':')[-1])
    else:
        crs_ = crs.from_string(crs_)
    return crs_


if __name__ == '__main__':
    args = get_parser().parse_args()
    # args = get_parser().parse_args("-i /data/bangalore/raster/WV3_2016-11_north.tif "
    # args = get_parser().parse_args("-i /data/bangalore/raster/WV3_Pansharpen_11_2016_subset.tif "
    #                                "-o /home/max/dr/deleteme.sqlite "
    #                                "-m /home/max/dr/models/jitted/bengaluru.pt "
    #                                "-w 512 "
    #                                "--ndvi "
    #                                "--red 3 "
    #                                "--nir 4".split())
    # args = get_parser().parse_args("-i /data/gartow/raster/tiles_buffered/gartow_T_do05_mosaic.vrt "
    #                                "-o /home/max/dr/deleteme.sqlite "
    #                                "-m /home/max/dr/models/jitted/gartow.pt "
    #                                "-w 1024 "
    #                                "--ndvi "
    #                                "--red 0 "
    #                                "--nir 3 "
    #                                "-s 0.1 -b 0.1 -l 0.1 --sigma 5".split())

    polygon_extraction_params = {"min_dist"          : args.min_dist,
                                 "mask_exp"          : 2,
                                 "outline_multiplier": 5,
                                 "dist_exp"          : 0.5,
                                 "sigma"             : args.sigma,
                                 "label_threshold"   : args.label_threshold,
                                 "binary_threshold"  : args.binary_threshold,
                                 "simplify"          : args.simplify
                                 }

    print("Loading model")
    
    model_names = args.model
    
    if isinstance(model_names, str):
        model = torch.jit.load(args.model).to(args.device)
    elif isinstance(model_names, list):
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

    model = InferenceModel(model)  # apply sigmoid to mask and outlines

    model = DataParallel(model)
    model.eval()

    array = xr.open_rasterio(args.input_file)
    nbands, height, width = array.shape

    if args.tilesize == 0:
        free_mem_bytes = psutil.virtual_memory().available
        byes_per_pixel = nbands * 4  # float32
        total_pixels = free_mem_bytes / byes_per_pixel / 8
        chunk_size = min(max(width, height), int(np.sqrt(total_pixels)))
    else:
        chunk_size = args.tilesize

    nchunks_h = len(range(0, height, chunk_size))
    nchunks_w = len(range(0, width, chunk_size))
    nchunks = nchunks_h * nchunks_w
    print("Chunk size for processing: {} pixels".format(chunk_size))

    #%%
    print("Starting processing...")

    polygons = []

    inference_time = 0
    postprocessing_time = 0
    disk_loading_time = 0

    t0 = time()

    try:
        for i, y in enumerate(range(0, height, chunk_size)):
            for j, x in enumerate(range(0, width, chunk_size)):
                idx = i * nchunks_w + j + 1
                print("Loading chunk {}/{}".format(idx, nchunks))
                t1 = time()
                chunk = array[:, y:y + chunk_size, x:x + chunk_size].load()  #.transpose('y', 'x', 'band')
                data = chunk.data
                if args.divisor != 1:
                    data = data / args.divisor

                if args.ndvi:
                    # data = np.concatenate((data, ndvi(data, red=args.red, nir=args.nir, axis=2)[...,None]), axis=2)
                    n = ndvi(data, red=args.red, nir=args.nir, axis=0)[None, ...]
                    if args.rescale_ndvi:
                        n = (n + 1) / 2
                    data = np.concatenate((data, n), axis=0)

                t2 = time()
                disk_loading_time += t2 - t1
                print("Starting prediction on chunk {}/{}".format(idx, nchunks))
                result, (ymin, ymax, xmin, xmax) = utils.predict_on_array_cf(model,
                                                                             data,
                                                                             in_shape=(nbands + args.ndvi, args.width,
                                                                                       args.width),
                                                                             out_bands=3,
                                                                             drop_border=16,
                                                                             batchsize=args.batchsize,
                                                                             stride=stride,
                                                                             device=args.device,
                                                                             augmentation=args.augmentation,
                                                                             no_data=0,
                                                                             verbose=True,
                                                                             return_data_region=True)

                t3 = time()
                inference_time += t3 - t2

                if np.all(np.array((ymin, ymax, xmin, xmax)) == 0):
                    # in this case the prediction area was all no data, nothing to extract from here
                    print("Empty chunk, skipping remaining steps.")
                    continue

                print("Prediction done, extracting polygons for chunk {}/{}.".format(idx, nchunks))

                if args.save_prediction is not None:
                    utils.array_to_tif(result.transpose(1, 2, 0),
                                       args.save_prediction + "_{}-{}.tif".format(y, x),
                                       transform=utils.xarray_trafo_to_gdal_trafo(chunk.attrs["transform"]),
                                       crs=array.attrs["crs"])

                t4 = time()

                if args.subsample:
                    xres, xskew, xr, yskew, yres, yr = utils.get_xarray_trafo(chunk[:, ymin:ymax, xmin:xmax])
                    xres *= 2
                    yres *= 2
                    trafo = (xres, xskew, xr, yskew, yres, yr)
                    args.sigma /= 2
                    args.min_dist /= 2

                    polygons.extend(extract_polygons(*result[:, ymin:ymax:2, xmin:xmax:2],
                                                     transform=trafo,
                                                     area_min=3,
                                                     **polygon_extraction_params))

                else:
                    trafo = utils.get_xarray_trafo(chunk[:, ymin:ymax, xmin:xmax])
                    polygons.extend(extract_polygons(*result[:, ymin:ymax, xmin:xmax],
                                                     transform=trafo,
                                                     area_min=3,
                                                     **polygon_extraction_params))
                t5 = time()
                postprocessing_time += t5 - t4

        print("Found {} polygons in total.".format(len(polygons)))
        print("Total processing time: {}s".format(int(time() - t0)))
        print("Time loading from disk: {}s".format(int(disk_loading_time)))
        print("Inference time: {}s".format(int(inference_time)))
        print("Post-processing time: {}s".format(int(postprocessing_time)))
        print("Saving as {}".format(args.output_file))

        crs_ = get_crs(array)

        utils.save_polygons(polygons,
                            args.output_file,
                            crs=crs_)
        print("Done.")

    except (MemoryError, KeyboardInterrupt, ValueError) as e:
        traceback.print_exc()

        print("Found {} polygons in total.".format(len(polygons)))
        print("Total processing time: {}s".format(int(time() - t0)))
        print("Time loading from disk: {}s".format(int(disk_loading_time)))
        print("Inference time: {}s".format(int(inference_time)))
        print("Post-processing time: {}s".format(int(postprocessing_time)))
        print("Saving as {}".format(args.output_file))
        utils.save_polygons(polygons,
                            args.output_file,
                            crs=get_crs(array))
        print("Errors were encountered during processing, see above. Polygons found so far have been saved.")

    except fiona.errors.CRSError as e:
        traceback.print_exc()

        fname = args.output_file.split('.')[0] + '.pickle'

        print("Total processing time: {}s".format(int(time() - t0)))
        print("Time loading from disk: {}s".format(int(disk_loading_time)))
        print("Inference time: {}s".format(int(inference_time)))
        print("Post-processing time: {}s".format(int(postprocessing_time)))

        print("Error during saving the polygons. Saving as 'pickled' python object here:\n{}".format(fname))
        print("You can use pickle to load them and try to save manually.")

        with open(fname, 'wb') as f:
            pickle.dump(polygons, f)
