#!/usr/bin/env python
import os
import gc
import sys
import fiona
import xarray as xr
import rioxarray  # needed for xarray export to tif, dont remove
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from osgeo import osr, ogr, gdal
from osgeo import gdalnumeric as gdn
from shapely.geometry import shape, Polygon
from multiprocessing import Pool, Lock


gdal.UseExceptions()  # This is fucking important. Only god knows why it's disabled by default.
fiona.drvsupport.supported_drivers["SQLite"] = "r"


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
    xres, yres = (arr.transform[0], arr.transform[4])
    xskew, yskew = (arr.transform[1], arr.transform[3])
    return xres, xskew, min(xr), yskew, yres, max(yr)


def get_xarray_extent(arr):
    """Returns
    xmin, xmax, ymin, ymax, xres, yres
    of an xarray. xres and yres can be negative.
    """
    xr = arr.coords["x"].data
    yr = arr.coords["y"].data
    xres, yres = (arr.transform[0], arr.transform[4])
    return min(xr), max(xr), min(yr), max(yr), xres, yres


def extent_to_poly(xarr):
    """Returns the bounding box of an xarray as shapely polygon."""
    xmin, xmax, ymin, ymax, xres, yres = get_xarray_extent(xarr)
    return Polygon([(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)])


def rasterize(source_raster, features: list, dim_ordering: str = "HWC"):
    """ Rasterizes the features (polygons/lines) within the extent of the given xarray with the same resolution, all in-memory.

    Args:
        source_raster: Xarray
        features: List of shapely objects
        dim_ordering: One of CHW (default) or HWC (height, widht, channels)
    Returns:
        Rasterized features
    """
    ncol = source_raster.sizes["x"]
    nrow = source_raster.sizes["y"]

    # Fetch projection and extent
    if "crs" in source_raster.attrs:
        proj = source_raster.attrs["crs"]
    else:
        proj = source_raster.rio.crs.to_proj4()

    ext = xarray_trafo_to_gdal_trafo(get_xarray_trafo(source_raster))

    raster_driver = gdal.GetDriverByName("MEM")
    out_raster_ds = raster_driver.Create('', ncol, nrow, 1, gdal.GDT_Byte)
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    spatref = osr.SpatialReference()
    spatref.ImportFromProj4(proj)

    vector_driver = ogr.GetDriverByName("Memory")
    vector_ds = vector_driver.CreateDataSource("")
    vector_layer = vector_ds.CreateLayer("", spatref, ogr.wkbMultiLineString)
    defn = vector_layer.GetLayerDefn()

    for poly in features:
        feature = ogr.Feature(defn)
        geom = ogr.CreateGeometryFromWkb(poly.wkb)
        feature.SetGeometry(geom)
        vector_layer.CreateFeature(feature)

    vector_layer.SyncToDisk()

    gdal.RasterizeLayer(out_raster_ds,
                        [1],
                        vector_ds.GetLayer(),
                        burn_values=[1],
                        options=['ALL_TOUCHED=TRUE']
                        )

    out_raster_ds.FlushCache()
    bands = [out_raster_ds.GetRasterBand(i) for i in range(1, out_raster_ds.RasterCount + 1)]
    # arr = xr.zeros_like(source_raster[[0],:,:])
    arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(np.uint8)
    if dim_ordering == "HWC":
        arr = arr.transpose((1, 2, 0))
    del out_raster_ds
    del vector_ds
    return arr


def clip_and_save(feature_enum, dest_fname_base, src_fname, src_xarray, lock, restrict_to_intersection=True, mask=False, no_overwrite=False):
    j, feature = feature_enum
    i = feature["id"] if "id" in feature else j
    src_bbox = extent_to_poly(src_xarray)
    dest_fname = dest_fname_base + "_{}.tif".format(i)
    print(dest_fname)
    if os.path.exists(dest_fname) and no_overwrite:
        return

    feature_polygon = shape(feature["geometry"])
    intersection = feature_polygon.intersection(src_bbox)
    # double check
    if intersection.area > 0:
        if mask:
            if restrict_to_intersection:
                tmp = src_xarray.rio.clip_box(*intersection.bounds).load()
                mask_arr = rasterize(tmp, [feature_polygon], dim_ordering="CHW")
                result = tmp * mask_arr
                del tmp

            else:
                mask_arr = rasterize(src_xarray, [feature_polygon], dim_ordering="CHW")
                result = src_xarray * mask_arr

            result.rio.to_raster(dest_fname,
                                 driver="GTiff",
                                 compress="DEFLATE",
                                 alpha="no")
            del mask_arr
            del result

        else:
            if restrict_to_intersection:
                bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = intersection.bounds
            else:
                bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = feature_polygon.bounds
            options = gdal.TranslateOptions(projWin=[bbox_min_x, bbox_max_y, bbox_max_x, bbox_min_y], creationOptions=['COMPRESS=DEFLATE'])
            try:
                gdal.Translate(dest_fname, src_fname, options=options)
            except Exception as e:
                print("[WARNING] Could not process file {}, feature {}.".format(src_fname, i))
                print(e)
    gc.collect()  # please save me


def get_parser():
    parser = ArgumentParser(description="Loads a raster and a shapefile, then produces rectangular tif tiles\n"
                                        "covering each shape in the shapefile. You can mask the resulting arrays\n"
                                        "with the overlay polygon.\n"
                                        "WARNING: With lots of processes and large VRT files, the RAM gets slowly\n"
                                        "filled by the underlying read cache. You can simply kill the process and\n"
                                        "restart with --no-overwrite.\n"
                                        "The algorithm is parallelized across shapefile features, not input files.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_file",
                        dest="input_files",
                        type=str,
                        nargs='*',
                        help="Input raster file",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output",
                        type=str,
                        help="Output file name pattern. For example '~/output/tile_' will use 'tile_' as "
                             "prefix and writes everything into the output folder.",
                        required=True)
    parser.add_argument("-shp", "--shapefile",
                        dest="shpfile",
                        type=str,
                        help="Shapefile to be iterated over (eg ESRI, SQLite).",
                        required=True)
    parser.add_argument("-r", "--restrict",
                        dest="r",
                        action='store_true',
                        help="If given, the resulting rasters are restricted to the area of "
                             "intersection between source raster and feature.")
    parser.add_argument("-m", "--mask",
                        dest="m",
                        action='store_true',
                        help="If given, only the values within the polygons are extracted. The rest is set to zero.")
    parser.add_argument("--no-overwrite",
                        dest="no_overwrite",
                        action='store_true',
                        help="If given, data will not be overwritten. The default is to overwrite.")
    parser.add_argument("-p", "--processes",
                        dest="nprocs",
                        type=int,
                        default=1,
                        help="Number of processes to start.",
                        required=False)
    parser.add_argument("--chunksize",
                        dest="chunksize",
                        type=int,
                        default=1,
                        help="Controls how many features each process handles at once.",
                        required=False)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    input_files = [os.path.abspath(f) for f in args.input_files]
    input_path, _ = os.path.split(input_files[0])
    output_pattern = os.path.abspath(args.output)
    shapefile = os.path.abspath(args.shpfile)
    shp_format = shapefile.split('.')[-1]

    if not os.path.isfile(shapefile):
        print("File '{}' not found.".format(shapefile))
        sys.exit(1)

    if not os.path.exists(os.path.abspath(os.path.join(output_pattern, os.pardir))):
        print("Output folder does not exist. Please create it.")
        sys.exit(1)

    print("Input raster files: ", len(input_files))

    for i, file_ in enumerate(input_files):

        if not os.path.isfile(file_):
            print("File '{}' not found.".format(file_))
            sys.exit(1)

        _, filename = os.path.split(file_)
        filename = filename.split('.')[0]
        dest_fname_base = output_pattern + filename

        readlock = Lock()
        source_raster = xr.open_rasterio(file_, "r", lock=readlock, cache=False)
        source_raster_bbox = extent_to_poly(source_raster)

        def work(feature_enum):
            clip_and_save(feature_enum,
                          dest_fname_base,
                          file_,
                          source_raster,
                          None,
                          restrict_to_intersection=args.r,
                          mask=args.m,
                          no_overwrite=args.no_overwrite)

        with Pool(args.nprocs) as p:
            with fiona.open(shapefile) as src:
                print("Feature count:", len(src))
                p.map(work, enumerate(src.filter(bbox=source_raster_bbox.bounds)), chunksize=args.chunksize)
