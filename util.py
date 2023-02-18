import numpy as np
import torch


def iou(pred, target, n_classes:int=21) -> float:
    """
    Compute the Intersection-over-Union (IoU) of a single pattern, where
    for some object class k:

    pixels(pattern, k) = set of pixels in the given pattern labeled class k

    IoU of k = intersect(pixels(pred, k), pixels(target, k)) / union(pixels(pred, k), pixels(target, k))

    After computing the IoU for each class, we simply average over each class to
    obtain a general IoU for the entire pattern.

    :param pred: array containing the pixel predictions of the model
    :param target: array containing the class labels of the input
    :param n_classes: The expected number of classes in the pattern
    :return: The IoU of the pattern as a float
    """
    # pre-process pred and target, some entries will have value 255,
    # denoting object boundaries. set to 0, denoting background class
    pred[pred == 255] = 0  # TODO will our model even output object boundary pixels? this may be unnecessary
    target[pred == 255] = 0
    # num pixels is equal to num entries in target and/or pred
    total = target.size().numel()
    class_iou = np.zeros(n_classes)
    for k in range(n_classes):
        # given the set of pixels in pred and target
        # labeled k, we want to calculate the size of
        # their intersection and union.
        normalized_pred = pred - k
        normalized_target = target - k
        labeled_pred = total - np.count_nonzero(normalized_pred)
        labeled_target = total - np.count_nonzero(normalized_target)
        # intersection is number of pixels that are
        # both labeled 0 in normalized
        # logical or, true when either is non zero, false when both are zero
        # count nonzero counts trues
        intersect = total - np.count_nonzero(normalized_target | normalized_pred)
        # basic combinatorics, size of two sets is equal
        # to size A + size B - intersection
        union = 1 + labeled_target + labeled_pred - intersect
        class_iou[k] = intersect / union
    return np.average(class_iou)


def pixel_acc(pred, target) -> float:
    """
    Compute the pixel accuracy (PA) of a single pattern, where:

    PA = # pixels w/ correct predictions / # of pixels

    :param pred: array containing the pixel predictions of the model
    :param target: array containing the class labels of the input
    :return: A float, denoting PA according to calculation above
    """
    # pre-process pred and target, some entries will have value 255,
    # denoting object boundaries. set to 0, denoting background class
    pred[pred == 255] = 0 # TODO will our model even output object boundary pixels? this may be unnecessary
    target[pred == 255] = 0
    # num pixels is equal to num entries in target and/or pred
    total = target.size().numel()
    # calc number of true positive predictions
    # subtract pred from target, if 2 pixels have same
    # class label, then difference pixel will be 0. Otherwise
    # difference will be nonzero.
    diff = pred - target
    # count zeros in diff to get number of true positives
    num_correct = total - np.count_nonzero(diff)
    return num_correct / total
