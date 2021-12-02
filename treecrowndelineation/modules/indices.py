def ndvi_xarray(img, red, nir):
    """Calculates the NDVI from a given image. Implicitly converts to Float32."""
    redl = img.sel(band=red).astype('float32')
    nirl = img.sel(band=nir).astype('float32')
    return (nirl - redl) / (nirl + redl + 1E-10)


def ndvi(img, red, nir, axis=-1):
    """Calculates the NDVI from a given image. Implicitly converts to Float32."""
    if axis == -1 or axis == 2:
        redl = img[:, :, red].astype('float32')
        nirl = img[:, :, nir].astype('float32')
    elif axis == 0:
        redl = img[red].astype('float32')
        nirl = img[nir].astype('float32')
    else:
        raise ValueError("Calculating NDVI along axis {} not supported.".format(axis))
    return (nirl - redl) / (nirl + redl + 1E-10)
