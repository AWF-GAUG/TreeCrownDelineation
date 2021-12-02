import numpy as np
from shapely.strtree import STRtree

from . import utils
from . import postprocessing as pp


def tp_fp_fn_polygons_indices(y_pred, y_true, iou_threshold=0.5):
    """Calculates true positives, false positives and false negatives among predicted polygons.

    Builds shapely.STRtrees internally.

    Args:
        y_pred: List of predicted shapely polygons
        y_true: List of true shapely polygons
        iou_threshold: Polygons which overlap with an iou of more than this will be counted as true positives

    Returns:
        Three lists of integers with the polygon indices which can be used to index into y_pred and y_true: (tp, fp, fn)
        True positives come from y_pred.
    """
    # y_pred_tree = STRtree(y_pred)
    y_true_tree = STRtree(y_true)
    tp = []
    fp = []
    fn = []

    for i, p_pred in enumerate(y_pred):
        if not p_pred.is_valid: continue  # If a polygon is not valid for some reason, we just skip & don't count it.
        tp_found = False
        intersecting_polygons = y_true_tree.query(p_pred)

        for p_inter in intersecting_polygons:
            intersection = p_pred.intersection(p_inter).area
            union = p_pred.union(p_inter).area  # we divide by the union, but as the two intersect it should be >0
            if intersection / union > iou_threshold:
                # in this case it's a true positive
                tp.append(i)
                tp_found = True
                break
        # if positives_found > 1:
        #     raise RuntimeError("More than one matching polygon found in y_true for poygon {} in y_pred".format(i))
        if not tp_found:
            fp.append(i)

    # let's find all the false negatives; those that are in y_true, but not in y_pred
    # we already know which *predicted* polygons are true positives, so all y_true polygons, which are not in tp
    # are necessarily fn; we can ignore all false positives
    rest = [idx for idx in range(len(y_pred)) if idx in tp]
    tp_tree = STRtree([y_pred[i] for i in rest])
    for i, p_true in enumerate(y_true):
        # we assume y_true only contains valid polygons
        intersecting_polygons = tp_tree.query(p_true)
        tp_found = False
        for p_inter in intersecting_polygons:
            intersection = p_true.intersection(p_inter).area
            union = p_true.union(p_inter).area
            if intersection / union > iou_threshold:
                # the polygon is a true positive, therefore
                tp_found = True
                break
        if not tp_found:
            fn.append(i)

    return tp, fp, fn


def tp_fp_fn_polygons_counts(y_pred, y_true, iou_threshold=0.5):
    """Calculates true positives, false positives and false negatives among predicted polygons.

    Builds a shapely.STRtree of the ground truth polygons internally.

    Args:
        y_pred: List of predicted shapely polygons
        y_true: List of true shapely polygons
        iou_threshold: Polygons which overlap with an iou of more than this will be counted as true positives

    Returns:
        Counts of tp, fp, fn
    """
    if iou_threshold < 0.5:
        raise ValueError(
            "The iou threshold must be above 0.5, not {}, otherwise the algorithm is not well defined.".format(
                iou_threshold))

    y_true_tree = STRtree(y_true)
    tp = 0
    fp = 0

    for i, p_pred in enumerate(y_pred):
        if not p_pred.is_valid: continue  # If a polygon is not valid for some reason, we just skip & don't count it.
        tp_found = False
        intersecting_polygons = y_true_tree.query(p_pred)

        for p_inter in intersecting_polygons:
            intersection = p_pred.intersection(p_inter).area
            union = p_pred.union(p_inter).area  # we divide by the union, but as the two intersect it should be >0
            if intersection / union > iou_threshold:
                # in this case it's a true positive
                tp += 1
                tp_found = True
                break
        if not tp_found:
            fp += 1

    fn = len(y_true) - tp
    return tp, fp, fn


def tp_fp_fn_polygons_from_list(predictions: list, true_polygons: list, rasters, iou_threshold=0.5, **kwargs) -> \
        tuple:
    """ Calculates the true positive, false positive and false negative polygons from a list of predictions.

    Args:
        predictions (list): List of predictions which contain masks, outlines and the distance transform stacked along
            dimension 3.
        true_polygons (list): List of lists of true shapely polygons. The order of sublists must match the predictions.
        rasters: A dataset containing the rasters from which the predictions were made. Needed to georeference the
            resulting poylgons.
        kwargs: The arguments for the underlying polygon extraction: amin, amax mask_exp, contour_multiplier,
            contour_exp, sigma, threshold, min_dist. See postprocessing.find_treecrowns() for reference.

    Returns:
        A tuple containing three sets of polygons: tp, fp, fn
    """
    tp, fp, fn = [], [], []
    for i, pred in enumerate(predictions):
        mask = pred[:, :, 0]
        outline = pred[:, :, 1]
        dist = pred[:, :, 2]
        # xmin, xmax, ymin, ymax, xres, yres = utils.get_xarray_extent(dataset.rasters[i])
        trafo = utils.get_xarray_trafo(rasters[i])

        found_polygons = np.array(pp.extract_polygons(mask, outline, dist, trafo, **kwargs))
        tp_, fp_, fn_ = tp_fp_fn_polygons_indices(found_polygons, true_polygons[i], iou_threshold)
        tp.extend(np.array(found_polygons)[tp_])
        fp.extend(np.array(found_polygons)[fp_])
        fn.extend(np.array(true_polygons[i])[fn_])

    return tp, fp, fn


def iou_matrix_naive(y_pred, y_true):
    # takes ages, just for validation of other code
    res = np.zeros((len(y_pred), len(y_true)))
    for i, p_pred in enumerate(y_pred):
        for j, p_true in enumerate(y_true):
            intersection = p_pred.intersection(p_true).area
            union = p_pred.union(p_true).area + 1E-8
            iou = intersection / union
            res[i, j] = iou
    return res


def oversegmentation_factor(y_true: list, y_pred: STRtree, threshold=0.5):
    overlapping_polygons = 0
    for p_true in y_true:
        intersecting_polygons = y_pred.query(p_true)  # these polygons have an overlapping bounding box with p
        for p_inter in intersecting_polygons:
            a = p_inter.area
            intersection = p_true.intersection(p_inter).area
            if intersection / a > threshold:
                overlapping_polygons += 1

    if len(y_true) > 0:
        return overlapping_polygons / len(y_true)
    else:
        return 0
