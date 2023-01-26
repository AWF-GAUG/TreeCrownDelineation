#! /usr/bin/env python
import os
import numpy as np
from glob import glob

from osgeo import gdal, osr, ogr
from osgeo import gdalnumeric as gdn

# import shapely
import fiona
import rioxarray  # don't comment this out; it modifies the xarray class for saving geotiffs
import xarray as xr
from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree

from itertools import islice, repeat
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import sobel, convolve
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from matplotlib import pyplot as plt
from skimage.morphology import dilation, square

from multiprocessing.pool import Pool, ThreadPool

fiona.drvsupport.supported_drivers["SQLite"] = "rw"

def get_parser():
    parser = ArgumentParser(description="Takes a list of rasters and rasterizes shapely polygons on them."
                                        "The polygons are distance-transformed and normalized on a per-polygon basis "
                                        "to the maximum distance.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_file",
                        dest="input_files",
                        type=str,
                        nargs='+',
                        help="Input raster file(s). If the file name is given in the form name_id.xyz the underscore-\
                             separated id can optionally be used to infer the output raster names by giving --infer_names.",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output",
                        type=str,
                        help="Output file name pattern. For example '~/output/mask_' will use 'mask_' as  \
                             prefix and writes everything into the output folder.",
                        required=True)
    parser.add_argument("-shp", "--shapefile",
                        dest="shapefile",
                        type=str,
                        help="Shapefile to be iterated over.",
                        required=True)
    parser.add_argument("-t",
                        dest="threads",
                        type=int,
                        help="Number of parallel threads to execute.",
                        default=1)
    parser.add_argument("-amax",
                        dest="area_max",
                        help="Maximum polygon area in map units, e.g. m2",
                        default=None)
    parser.add_argument("-amin",
                        dest="area_min",
                        help="Minimum polygon area in map units, e.g. m2",
                        default=3)
    parser.add_argument("-ccn", "--class_col_name",
                        default=None,
                        help="Name of the class column. In conjunction with -cls this can be used to rasterize only "
                             "polygons of a certain class.")
    parser.add_argument("-cls",
                        default="1,",
                        help="Class ids to be processed. Can be a comma separated list: '1, 2, 3'")
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


def filter_geometry(src):
    return (shape(f["geometry"]) for f in src)


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
    arr = xr.zeros_like(source_raster[[0],:,:])
    arr[:] = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(np.uint8)
    arr.attrs["nodatavals"] = (0,)
    arr.attrs["scales"] = (1,)
    arr.attrs["offsets"] = (0,)

    if dim_ordering == "HWC":
        arr = arr.transpose(('y','x','band'))
    del out_raster_ds
    del vector_ds
    return arr


def fix_xarray_metadata(xarr, idxs):
    xarr.attrs["nodatavals"] = tuple(np.array(xarr.nodatavals)[idxs])
    xarr.attrs["scales"] = tuple(np.array(xarr.scales)[idxs])
    xarr.attrs["offsets"] = tuple(np.array(xarr.offsets)[idxs])
    # xarr.attrs["descriptions"] = tuple(np.array(xarr.descriptions)[idxs])


def to_outline(polygons):
    return (p.boundary for p in polygons)


def work(f, args):
    shpfile_path = args.shapefile
    area_max = args.area_max
    area_min = args.area_min

    # print(args.class_col_name)
    # print(args.cls.split(","))

    valid_classes = [int(i) for i in args.cls.split(",") if i not in ("None", "NULL", "")]
    if "None" in args.cls.split(",") or "NULL" in args.cls.split(","):
        valid_classes.append(None)
    # print(valid_classes)

    input_file = os.path.abspath(f)
    input_path, input_fname = os.path.split(input_file)
    suffix = input_fname.split('.')[0].split("_")[-1]

    output_file = os.path.abspath(args.output) + suffix + ".tif"
    print(output_file)

    img = rioxarray.open_rasterio(f)
    mask = xr.zeros_like(img[[0]]).astype("float32")  # dirty hack to get three layers
    # fix_xarray_metadata(mask, [0])

    # bbox = extent_to_poly(img)
    # features = strtree.query(bbox)

    # m = rasterize(img, features, dim_ordering="CHW")[0].astype(int)
    # outline = rasterize(img, to_outline(features), dim_ordering="CHW")[0]
    # print(outline.shape)
    # outline = dilation(outline, square(3))
    # m = m - outline
    # m = np.clip(m, 0, 1)

    # mask[0] = distance_transform_edt(m)
    # mask[1] = sobel(mask[0], axis=0, mode="constant")
    # mask[2] = sobel(mask[0], axis=1, mode="constant")
    #
    # mask[0] = mask[0] / np.max(mask[0])
    # norm = np.linalg.norm(mask[1:], axis=0) + 1E-5
    # mask[1:] /= norm

    xmin, xmax, ymin, ymax, _, _ = get_xarray_extent(img)
    with fiona.open(shpfile_path, "r") as features:
        haskey = args.class_col_name in features.schema["properties"].keys()
        if not haskey:
            print("Vector data has no class field {}, attempting anyway.".format(args.class_col_name))

        features = features.filter(bbox=(xmin, ymin, xmax, ymax))

        i = 0
        for f in features:
            i += 1
            if haskey:
                cls = f["properties"][args.class_col_name]
                # print(cls)
                if cls not in valid_classes:
                    print("skipping polygon of unmatched class with id ", f["id"])
                    continue

            p = shape(f["geometry"])
            if area_max is not None:
                if p.area > area_max:
                    print("skipping too large polygon with id  ", f["id"])
                    continue

            if p.area < area_min:
                print("skipping too small polygon with id  ", f["id"])
                continue

            xmin_p, ymin_p, xmax_p, ymax_p = p.bounds
            polygon_area = mask.loc[:, ymax_p:ymin_p, xmin_p:xmax_p].astype("float32")
            if 0 in polygon_area.shape:
                continue

            try:
                rasterized = rasterize(polygon_area, [p], dim_ordering="CHW")[0]
            except ValueError as err:
                print("errored at:", output_file)
                print("POLYGON INDEX", i)
                print(p)
                print(polygon_area)
                raise err

            padded = np.pad(rasterized,1)
            distance_transformed = distance_transform_edt(padded)[1:-1,1:-1].astype("float32")
            distance_transformed /= max(np.max(distance_transformed), 1)
            polygon_area[0] = distance_transformed
            mask.loc[:, ymax_p:ymin_p, xmin_p:xmax_p] += polygon_area
            # transformed_polygon = shapely.affinity.affine_transform(p, (1/xres, 0, 0, 1/yres, -xmin/xres, -ymax/yres))
            # plt.plot(*transformed_polygon[0].exterior.xy, "r")

    mask[0] = np.clip(mask[0], 0, 1)
    mask.rio.to_raster(output_file, compress="DEFLATE")

#%%
if __name__ == '__main__':
    args = get_parser().parse_args()
    # args = get_parser().parse_args("-i /data_hdd/bkg/training/tiles/tile_4026.tif -o /tmp/deleteme/dist_ -shp /data_hdd/bkg/training/training_labels_bkg_2022-02-15.sqlite".split())
    # args = get_parser().parse_args("-i /data/bangalore/training_data/treecrown_delineation/tiles/tile_WV3_Pansharpen_11_2016_9.tif -o ./asdf_ -shp /data/bangalore/training_data/treecrown_delineation/treecrown_delineation_north_2016.sqlite".split())
    # args = get_parser().parse_args("-i /home/max/dr/gartow/masks/mask_100.tif -o ./asdf_ -shp "
    #                                "/data/gartow/vector/polygons.sqlite".split())

    # with fiona.open(args.shapefile) as src:
    #     strtree = STRtree(filter_geometry(src))

    files = args.input_files
    with Pool(args.threads) as pool:
    #     # pool.starmap(work, zip(files, repeat(strtree, len(files))))
        pool.starmap(work, zip(files, repeat(args, len(files))))

    # t0 = time()
    # input_files = glob(args.input_files[0])
    # print(input_files)
    # for f in args.input_files:
    #     # work(f, strtree)
    #     work(f, args.shapefile)
    # print(time()-t0)

#%%
# from time import time
# from matplotlib import pyplot as plt
# plt.clf()
# plt.imshow((mask[[1,2,2],:1000,:1000].data.transpose((1,2,0))+1)/2)
# # plt.imshow(mask[1], vmin=-1, vmax=1)
# mask.rio.to_raster("./test.tif", compress="DEFLATE")
#%%

