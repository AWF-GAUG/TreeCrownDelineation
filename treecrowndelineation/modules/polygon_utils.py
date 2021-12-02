import numpy as np
import gdalnumeric as gdn
import xarray as xr
from osgeo import osr
from osgeo import ogr
from osgeo import gdal

from . import utils


def rasterize(source_raster: xr.DataArray, features: list, dim_ordering: str = "HWC"):
    """ Rasterizes the features (polygons/lines) within the extent of the given xarray with the same resolution, all in-memory.

    Args:
        source_raster: Xarray
        features: List of shapely objects
        dim_ordering: One of CHW or HWC (height, widht, channels) (default)
    Returns:
        Rasterized features as numpy array
    """
    # Fetch number of rows and columns
    ncol = source_raster.sizes["x"]
    nrow = source_raster.sizes["y"]

    # Fetch projection and extent
    if "crs" in source_raster.attrs:
        proj = source_raster.attrs["crs"]
    else:
        proj = source_raster.rio.crs.to_proj4()
    ext = utils.xarray_trafo_to_gdal_trafo(utils.get_xarray_trafo(source_raster))

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
    return arr
