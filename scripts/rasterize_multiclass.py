#!/usr/bin/env python
import os
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from osgeo import gdalnumeric as gdn

import rioxarray
import xarray as xr
import fiona
from shapely.geometry import shape, Polygon

from multiprocessing import Pool

fiona.drvsupport.supported_drivers["SQLite"] = "rw"

def get_parser():
    parser = ArgumentParser(description="Loads a raster and a vector file, then rasterizes the vector file within the \
                                         extent of the raster with the same resolution. Uses gdal_rasterize \
                                         under the hood, but provides some more features like specifying which classes \
                                         to rasterize into which layer of the output. If you want to infer the output \
                                         file names, the input file name suffixes have to be delimited by an '_'.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_file",
                        dest="input_files",
                        type=str,
                        nargs='+',
                        help="Input raster file. If the file name is given in the form name_id.xyz the underscore-\
                             separated id can optionally be used to infer the output raster names by giving --infer_names.",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output",
                        type=str,
                        help="Output file name pattern. For example '~/output/mask_' will use 'mask_' as  \
                             prefix and writes everything into the output folder.",
                        required=True)
    parser.add_argument("-shp", "--shapefile",
                        dest="shpfile",
                        type=str,
                        help="Shapefile to be iterated over.",
                        required=True)
    parser.add_argument("-p",
                        dest="nprocs",
                        type=int,
                        help="Number of parallel processes to execute.",
                        default=1)
    # parser.add_argument("-t", "--type_col_name",
    #                     dest="type_col_name",
    #                     type=str,
    #                     default=None,
    #                     help="Optionally, the name of the type/class column can be given, in order to be included in output file names.")
    # parser.add_argument("--fully_contained_only",
    #                     dest='fully_contained_only',
    #                     action='store_true',
    #                     help="If given, only polygons which are fully contained in the raster will be rasterized.")
    # parser.add_argument("--build-tree",
    #                     dest='build_tree',
    #                     action='store_true',
    #                     help="If given, an STR-tree is built from the given shapefile to speed up computation "
    #                          "for high numbers of features.")
    parser.add_argument("--outlines",
                        dest='outlines',
                        action='store_true',
                        help="If given, the outlines of the given features will be rasterized.")

    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-cls",
                        default=None,
                        help="The valid class ids to be rasterized. Can be a single integer or a list of integers like "
                             "so: '1, 2, 3'")
    parser.add_argument("-ccn", "--class_col_name",
                        default=None,
                        help="Name of the class column. In conjunction with -cls this can be used to rasterize only "
                             "polygons of a certain class.")
    # parser.add_argument("--num_classes",
    #                    dest="num_classes",
    #                    type=int,
    #                    help="The number of classes to rasterize. All objects of class_id <= num_classes will be \
    #                         rasterized. One of --num_classes or --mapping has to be given.")
    # parser.add_argument("--mapping",
    #                    dest="mapping",
    #                    type=str,
    #                    nargs='+',
    #                    help="Defines which class is rasterized into which layer. Has to be given in the form \
    #                    'class1 layer1 class2 layer1 class4 layer2...' separated by spaces. \
    #                    Layer indexing starts with 1. \
    #                    One of --num_classes or --mapping has to be given.")

    return parser


def xarray_trafo_to_gdal_trafo(xarray_trafo):
    xres, xskew, xmin, yskew, yres, ymax = xarray_trafo
    return (xmin, xres, xskew, ymax, yskew, yres)


def get_xarray_trafo(arr):
    """Returns
    xmin, xmax, ymin, ymax, xres, yres
    of an xarray. xres and yres can be negative.
    """
    xr = arr.coords["x"].data
    yr = arr.coords["y"].data
    gt = [float(x) for x in arr.spatial_ref.GeoTransform.split()]
    xres, yres = (gt[1], gt[5])
    xskew, yskew = (gt[2], gt[4])
    return xres, xskew, min(xr), yskew, yres, max(yr)


def get_xarray_extent(arr):
    """Returns
    xmin, xmax, ymin, ymax, xres, yres
    of an xarray. xres and yres can be negative.
    """
    xr = arr.coords["x"].data
    yr = arr.coords["y"].data
    gt = [float(x) for x in arr.spatial_ref.GeoTransform.split()]
    xres, yres = (gt[1], gt[5])
    return min(xr), max(xr), min(yr), max(yr), xres, yres


def extent_to_poly(xarr):
    """Returns the bounding box of an xarray as shapely polygon."""
    xmin, xmax, ymin, ymax, xres, yres = get_xarray_extent(xarr)
    return Polygon([(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)])


# def rasterize(source_raster, features: list, dim_ordering: str = "HWC"):
#     """ Rasterizes the features (polygons/lines) within the extent of the given xarray with the same resolution, all in-memory.

#     Args:
#         source_raster: Xarray
#         features: List of shapely objects
#         dim_ordering: One of CHW (default) or HWC (height, widht, channels)
#     Returns:
#         Rasterized features
#     """
#     ncol = source_raster.sizes["x"]
#     nrow = source_raster.sizes["y"]

#     # Fetch projection and extent
#     if "crs" in source_raster.attrs:
#         proj = source_raster.attrs["crs"]
#     else:
#         proj = source_raster.rio.crs.to_proj4()

#     ext = xarray_trafo_to_gdal_trafo(get_xarray_trafo(source_raster))

#     raster_driver = gdal.GetDriverByName("MEM")
#     out_raster_ds = raster_driver.Create('', ncol, nrow, 1, gdal.GDT_Byte)
#     out_raster_ds.SetProjection(proj)
#     out_raster_ds.SetGeoTransform(ext)

#     spatref = osr.SpatialReference()
#     spatref.ImportFromProj4(proj)

#     vector_driver = ogr.GetDriverByName("Memory")
#     vector_ds = vector_driver.CreateDataSource("")
#     vector_layer = vector_ds.CreateLayer("", spatref, ogr.wkbMultiLineString)
#     defn = vector_layer.GetLayerDefn()

#     for poly in features:
#         feature = ogr.Feature(defn)
#         geom = ogr.CreateGeometryFromWkb(poly.wkb)
#         feature.SetGeometry(geom)
#         vector_layer.CreateFeature(feature)

#     vector_layer.SyncToDisk()

#     gdal.RasterizeLayer(out_raster_ds,
#                         [1],
#                         vector_ds.GetLayer(),
#                         burn_values=[1],
#                         options=['ALL_TOUCHED=TRUE']
#                         )

#     out_raster_ds.FlushCache()
#     bands = [out_raster_ds.GetRasterBand(i) for i in range(1, out_raster_ds.RasterCount + 1)]
#     arr = xr.zeros_like(source_raster[[0],:,:])
#     arr[:] = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(np.uint8)
#     arr.attrs["nodatavals"] = (0,)
#     arr.attrs["scales"] = (1,)
#     arr.attrs["offsets"] = (0,)

#     if dim_ordering == "HWC":
#         arr = arr.transpose((1, 2, 0))
#     del out_raster_ds
#     del vector_ds
#     return arr


def rasterize(clip_raster, shp, class_col_name='class', num_classes=None, class_dict=None, fully_contained_only=False):
    """ Takes an ESRI-shapefile and rasterizes it within the boundaries of a given raster.

    Args:
        clip_raster (str): Path to raster
        shp (str): Path to shapefile
        class_col_name (str): Name of the column to be rasterized.
        num_classes (int): Number of classes. This can only be used, if classes are consecutively numbered and start with 1!
        class_dict (dict): Dictionary in the form class: layer, where layer starts from 0.
        fully_contained_only (bool): Give True, if only fully contained polygons should be rasterized.

    Returns:
        Binary ndarray containing the masks.
    """

    assert class_dict != num_classes, 'Please give class dict or number of classes.'

    class_dict = class_dict or {i: i for i in range(1, num_classes + 1)}
    num_classes = num_classes or len(class_dict)
    raster_ds = gdal.Open(clip_raster)
    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    xmin, ymax = ext[0], ext[3]
    xmax = xmin + ext[1] * ncol
    ymin = ymax - ext[5] * nrow

    spatref = osr.SpatialReference()
    spatref.ImportFromWkt(raster_ds.GetProjection())

    raster_ds = None

    memory_driver = gdal.GetDriverByName('MEM')
    out_raster_ds = memory_driver.Create('', ncol, nrow, num_classes, gdal.GDT_Byte)

    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    mb_v = ogr.Open(shp)
    mb_l = mb_v.GetLayer()

    if fully_contained_only:
        eps = 1E-7
        ring = ogr.Geometry(ogr.wkbLineString)
        ring.AddPoint(xmin - xmin * eps, ymin - ymin * eps)
        ring.AddPoint(xmin - xmin * eps, ymax + ymax * eps)
        ring.AddPoint(xmax - xmax * eps, ymax + ymax * eps)
        ring.AddPoint(xmax + xmax * eps, ymin - ymin * eps)
        ring.CloseRings()

        driver = ogr.GetDriverByName('MEMORY')
        tmpshpfn = uuid4().hex
        tmpds = driver.CreateDataSource(tmpshpfn)
        tmplyr = tmpds.CreateLayer(tmpshpfn, srs=spatref, geom_type=ogr.wkbPolygon)

        # 4 loop over features in initial shapefile and save intersections with ring to tmplyr
        dsshp = ogr.Open(shp)
        lyrshp = dsshp.GetLayer(0)

        ft = lyrshp.GetNextFeature()
        while ft:
            geom = ft.GetGeometryRef()
            if ring.Intersects(geom):
                tmplyr.CreateFeature(ft.Clone())
            ft = lyrshp.GetNextFeature()

    # Rasterize the shapefile layer to our new dataset
    for cls, layer in class_dict.items():
        if type(cls) is str:
            mb_l.SetAttributeFilter("{}='{}'".format(class_col_name, cls))
        else:
            mb_l.SetAttributeFilter("{}={}".format(class_col_name, cls))

        gdal.RasterizeLayer(out_raster_ds,
                            [layer],
                            mb_l,
                            burn_values=[1]
                            )

    if fully_contained_only:
        gdal.RasterizeLayer(out_raster_ds,
                            np.arange(num_classes)+1,
                            tmplyr,
                            burn_values=num_classes*[0],
                            options=['ALL_TOUCHED=True']
                            )

    bands = [out_raster_ds.GetRasterBand(i) for i in range(1, out_raster_ds.RasterCount + 1)]
    arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(np.int8)
    arr = np.transpose(arr, [1, 2, 0])
    return arr


def filter_geometry(src, args):
    if args.cls is not None:
        valid_classes = [int(i) for i in args.cls.split(",")]
        return (shape(f["geometry"]) for f in src if f["properties"][args.class_col_name] in valid_classes)
    else:
        return (shape(f["geometry"]) for f in src)


def to_outline(polygons):
    return (p.boundary for p in polygons)

#%%
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.shpfile):
        print("File '{}' not found.".format(args.shpfile))
        sys.exit(1)

    def work(f):
        with fiona.open(args.shpfile) as src:
            input_file = os.path.abspath(f)
            input_path, input_fname = os.path.split(input_file)
            suffix = input_fname.split('.')[0].split('_')[-1]

            output_file = os.path.abspath(args.output) + suffix + '.tif'

            print(output_file)

            img = rioxarray.open_rasterio(f)
            bbox = extent_to_poly(img)
            features = src.filter(bbox=bbox.bounds)

            if args.outlines:
                res = rasterize(img, to_outline(filter_geometry(features, args)), dim_ordering="CHW")
            else:
                res = rasterize(img, filter_geometry(features, args), dim_ordering="CHW")
            res.rio.to_raster(output_file, compress="DEFLATE")

    with Pool(args.nprocs) as p:
        p.map(work, args.input_files)
