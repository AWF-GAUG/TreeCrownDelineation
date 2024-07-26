#!/usr/bin/env python
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import fiona
from multiprocessing import Pool

fiona.drvsupport.supported_drivers["SQLite"] = "rw"


def get_parser():
    parser = ArgumentParser(description="Loads multiple raster and one vector file, then rasterizes the vector file within the \
                                         extent of the rasters with the same resolution. Uses gdal_rasterize \
                                         under the hood, but provides some more features like specifying which classes \
                                         to rasterize with which value in the output. If you want to infer the output \
                                         file names, the input file name suffixes have to be delimited by an '_'.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_files",
                        dest="input_files",
                        type=str,
                        nargs='+',
                        help="Input raster file(s). If the file name is given in the form name_id.xyz, the underscore-\
                             separated id can optionally be used to infer the output raster names by giving --infer_names.",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output",
                        type=str,
                        help="Output file name pattern. For example '~/output/mask_' will use 'mask_' as  \
                             prefix and writes everything into the output folder.",
                        required=True)
    parser.add_argument("--shp", "--shapefile",
                        dest="shpfile",
                        type=str,
                        help="Shapefile to be iterated over.",
                        required=True)
    parser.add_argument("-p",
                        dest="nprocs",
                        type=int,
                        help="Number of parallel processes to execute.",
                        default=1)
    parser.add_argument("--outlines",
                        dest='outlines',
                        action='store_true',
                        help="If given, the outlines of the given features will be rasterized.")
    parser.add_argument("--ccn", "--class_col_name",
                        dest="class_col_name",
                        help="Name of the class column. In conjunction with --cls this can be used to rasterize only "
                             "polygons of a certain class.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--classes",
                       default=None,
                       help="The valid class ids to be rasterized. Can be a single integer or a list of integers like "
                            "so: '1, 2, 3'. The classes will be assigned increasing values in the mask, starting from 1.")
    group.add_argument("--mapping",
                       dest="mapping",
                       default=None,
                       type=str,
                       help="Defines which class is rasterized to which value. Has to be given in the form \
                       'class1 layer1 class2 layer1 class4 layer2...' separated by spaces. \
                       Layer indexing starts with 1. \
                       One of --num_classes or --mapping has to be given.")

    return parser


def rasterize(outfile: str, clip_raster: str, shp: str, class_col_name: str, class_dict: dict):
    """ Takes an ESRI-shapefile and rasterizes it within the boundaries of a given raster.

    Args:
        outfile (str): output file path
        clip_raster (str): Path to raster
        shp (str): Path to shapefile
        class_col_name (str): Name of the column to be rasterized.
        class_dict (dict): Dictionary in the form class: index, where index is the value you want to set the output
            raster to for the respective class
    """
    raster_ds = gdal.Open(clip_raster)
    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    spatref = osr.SpatialReference()
    spatref.ImportFromWkt(raster_ds.GetProjection())

    memory_driver = gdal.GetDriverByName('MEM')
    out_raster_ds = memory_driver.Create('', ncol, nrow, 1, gdal.GDT_Byte)

    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    mb_v = ogr.Open(shp)
    mb_l = mb_v.GetLayer()

    # Rasterize the shapefile layer to our new dataset
    for cls, index in class_dict.items():
        if type(cls) is str:
            mb_l.SetAttributeFilter("{}='{}'".format(class_col_name, cls))
        else:
            mb_l.SetAttributeFilter("{}={}".format(class_col_name, cls))

        gdal.RasterizeLayer(out_raster_ds,
                            [1],
                            mb_l,
                            burn_values=[index]
                            )

    out_drv = gdal.GetDriverByName("GTiff")
    out_drv.CreateCopy(outfile, out_raster_ds, 1, options=["COMPRESS=ZSTD", "PREDICTOR=2", "TILED=YES"])


#%%
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.mapping is not None:
        items = args.mapping.split()
        keys = items[::2]
        values = [int(i) for i in items[1::2]]
        mapping = {k: v for k, v in zip(keys, values)}
    elif args.classes is not None:
        classes = args.classes.split()
        mapping = {k: v for k, v in zip(classes, range(1, len(args.classes)+1))}

    if not os.path.isfile(args.shpfile):
        print("File '{}' not found.".format(args.shpfile))
        sys.exit(1)

    def work(f):
        input_file = os.path.abspath(f)
        input_path, input_fname = os.path.split(input_file)
        suffix = input_fname.split('.')[0].split('_')[-1]
        output_file = os.path.abspath(args.output) + suffix + '.tif'
        print(output_file)
        rasterize(output_file, f, args.shpfile, class_col_name=args.class_col_name, class_dict=mapping)

    with Pool(args.nprocs) as p:
        p.map(work, args.input_files)
