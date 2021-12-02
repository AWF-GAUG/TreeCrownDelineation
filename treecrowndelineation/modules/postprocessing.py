import numpy as np
from shapely.geometry import Polygon
from scipy import ndimage as ndi
from skimage import filters
from skimage.segmentation import watershed
from skimage.feature import corner_peaks
from hyperopt import fmin, tpe, hp, STATUS_OK
from rasterio.transform import IDENTITY
from rasterio.features import shapes

from . import utils
from . import polygon_metrics as pm


def find_treecrowns(mask, contour,
                    mask_exp=1,
                    outline_multiplier=6,
                    outline_exp=8,
                    sig=4,
                    min_dist=10,
                    binary_threshold=0.16,
                    label_threshold=0.5,
                    dist_exp=0.25
                    ):
    """ Takes a mask and a contour and returns a segmentation into labeled areas.

    Returns:
        An image (numpy array) where each found object's area is labeled by a different index.
    """
    res = np.clip(mask ** mask_exp - outline_multiplier * contour ** outline_exp, 0, 1)
    res = filters.gaussian(res, sigma=sig)
    res_checkpoint = np.copy(res)
    binary_img = res > binary_threshold
    distance = ndi.distance_transform_edt(binary_img)
    res = res * distance ** dist_exp
    local_max = corner_peaks(res, indices=False, min_distance=int(min_dist), threshold_abs=label_threshold,
                             exclude_border=False)
    markers = ndi.label(local_max)[0]
    labels = watershed(-res_checkpoint, markers, mask=binary_img)
    return labels


def find_treecrowns_from_dist_trafo(mask, outlines, dist_trafo,
                                    mask_exp=2,
                                    outline_multiplier=5,
                                    outline_exp=1,
                                    dist_exp=0.5,
                                    min_dist=10,
                                    sigma=1,
                                    binary_threshold=0.1,
                                    label_threshold=0.1,
                                    return_indices=False
                                    ):
    """ Takes a mask and a distance transform and returns a segmentation into labeled areas.

    Returns:
        An image (numpy array) where each found object's area is labeled by a different index.
    """
    # binary_img = mask > binary_threshold
    res = np.clip(mask ** mask_exp - outline_multiplier * outlines ** outline_exp, 0, 1) * np.clip(dist_trafo, 0,
                                                                                                   1) ** dist_exp
    res = filters.gaussian(res, sigma=sigma)
    binary_img = res > binary_threshold
    markers = corner_peaks(res, indices=False, min_distance=int(min_dist), threshold_abs=label_threshold,
                           exclude_border=False,
                           p_norm=2).astype(np.int32)
    ndi.label(markers, output=markers)
    labels = watershed(-res, markers, mask=binary_img, connectivity=2)

    if return_indices:
        indices = corner_peaks(res, indices=True, min_distance=int(min_dist), threshold_abs=label_threshold,
                               exclude_border=False,
                               p_norm=2).astype(np.int32)
        return labels, indices
    else:
        return labels


def extract_polygons(mask, outline, dist, transform=IDENTITY, area_min=3, area_max=10 ** 6, simplify=0.3, **kwargs):
    """Takes a mask and a contour and returns the found polygons."""
    # new_args = {k,v for k,v in kwargs.items() if k not in ("area_min")}
    labels = find_treecrowns_from_dist_trafo(mask, outline, dist, **kwargs)
    if type(labels) == tuple:
        labels, indices = labels

    shape_gen = shapes(labels, labels.astype(bool), transform=transform)  # "polygon producer"
    found_polygons = []
    for p in shape_gen:
        poly = Polygon(p[0]["coordinates"][0])
        if area_min < poly.area < area_max:  # area units depend on transformation
            found_polygons.append(poly)

    if not simplify:
        return found_polygons
    else:
        simplified_polygons = [p.simplify(simplify, preserve_topology=True) for p in found_polygons]
        return simplified_polygons


def objective(prediction, true_polygons, ds, kwargs):
    """Calculates the accuracy of the post-processing steps given a certain set of hyperparameters.
    This calculation is based on the raw network output and the ground truth polygons.

    Args:
        prediction: List of predictions in HWC layout
        true_polygons: List of shapely polygons
        ds: Dataset containing the rasters from which the predictions were made; needed to geotransform the resulting polygons
        kwargs: Hyperparameters of the polygon extraction - see optimization space below

    Returns:
        Dictionary containing: loss, tp, fp, fn and status
    """
    tp = fp = fn = 0
    # compare each tile individually
    for i, pred in enumerate(prediction):
        mask = pred[:, :, 0]
        contour = pred[:, :, 1]
        xmin, xmax, ymin, ymax, xres, yres = utils.get_xarray_extent(ds.rasters[i])

        # arg_rest = {k: v for k, v in kwargs.items() if k not in ("amin", "amax")}
        arg_rest = {k: v for k, v in kwargs.items() if k not in ("amin", "amax")}
        found_polygons = extract_polygons(mask, contour, xmin, ymax, xres, yres,
                                          area_min=kwargs["amin"],
                                          area_max=kwargs["amax"],
                                          **arg_rest)
        tp_, fp_, fn_ = pm.tp_fp_fn_polygons_counts(found_polygons, true_polygons[i])
        tp += tp_
        fp += fp_
        fn += fn_

    acc = tp / (tp + fp + fn)
    return {"loss": -acc, "tp": tp, "fp": fp, "fn": fn, "status": STATUS_OK}


def optimize_postprocessing(model, val_ds, test_ds, val_polygon_file, test_polygon_file, max_evals=100):
    y_validation = utils.load_filtered_polygons(val_polygon_file, val_ds.rasters)
    y_test = utils.load_filtered_polygons(test_polygon_file, test_ds.rasters,
                                          minimum_area=3)
    y_pred_val = [utils.predict_on_array(model, raster.data, (256, 256, 8), (224, 224, 2), drop_border=16, batchsize=16,
                                         stride=112, augmentation=True) for raster in val_ds.rasters]

    optimization_space = {"mask_exp"          : hp.uniform("mask_exp", 0, 5),
                          "contour_exp"       : hp.uniform("contour_exp", 0, 8),
                          "contour_multiplier": hp.uniform("contour_multiplier", 0, 8),
                          "dist_exp"          : hp.uniform("dist_exp", 0, 1),
                          "sigma"             : hp.uniformint("sigma", 0, 5),
                          "min_dist"          : hp.uniformint("min_dist", 3, 15),
                          "label_threshold"   : hp.uniform("label_threshold", 0, 1),
                          "binary_threshold"  : hp.uniform("binary_threshold", 0.1, 0.7),
                          "amin"              : hp.uniformint("amin", 0, 50 ** 2),
                          "amax"              : hp.uniformint("amax", 80 ** 2, 200 ** 2)
                          }

    def optimize_me(kwargs):
        return objective(y_pred_val, y_validation, val_ds, kwargs)

    # carry out the actual optimization
    best_args = fmin(optimize_me, optimization_space, algo=tpe.suggest, max_evals=max_evals)
    val_acc = -1 * optimize_me(best_args)["loss"]
    print("Accuracy on validation set:", val_acc)

    # predict on the test set
    y_pred_test = [utils.predict_on_array(model, raster.data, (256, 256, 8), (224, 224, 2), drop_border=16,
                                          batchsize=16, stride=56, augmentation=True) for raster in test_ds.rasters]

    test_res = objective(y_pred_test, y_test, test_ds, best_args)
    test_acc = -1 * test_res["loss"]
    print("Accuracy on test set:", test_acc)

    return {"hyperparams": best_args, **test_res}
