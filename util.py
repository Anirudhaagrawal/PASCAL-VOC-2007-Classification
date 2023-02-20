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
    pred[pred == 255] = 0
    target[target == 255] = 0
    # init empty array to hold IoU for each class
    iou = np.zeros(n_classes)
    # iterate through each class
    for k in range(n_classes):
        # calc the union of the pixels in pred and target labeled class k
        union = np.count_nonzero(np.logical_or(pred == k, target == k))
        union=union+1e-10
        # calc the intersection of the pixels in pred and target labeled class k
        intersect = np.count_nonzero(np.logical_and(pred == k, target == k))
        # calc the IoU for class k
        iou[k] = intersect / union
    # return the mean IoU of the pattern
    return np.average(iou)


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
    pred[pred == 255] = 0
    target[target == 255] = 0
    # calc the number of pixels in the pattern
    n_pixels = np.prod(pred.shape)
    # calc the number of pixels with correct predictions
    correct_pixels = np.count_nonzero(pred == target)
    # calc and return the pixel accuracy
    return correct_pixels / n_pixels
