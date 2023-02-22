import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# def iou(pred, target, n_classes: int = 21) -> float:
#     """
#     Compute the Intersection-over-Union (IoU) of a single pattern, where
#     for some object class k:
#
#     pixels(pattern, k) = set of pixels in the given pattern labeled class k
#
#     IoU of k = intersect(pixels(pred, k), pixels(target, k)) / union(pixels(pred, k), pixels(target, k))
#
#     After computing the IoU for each class, we simply average over each class to
#     obtain a general IoU for the entire pattern.
#
#     :param pred: array containing the pixel predictions of the model
#     :param target: array containing the class labels of the input
#     :param n_classes: The expected number of classes in the pattern
#     :return: The IoU of the pattern as a float
#     """
#     # pre-process pred and target, some entries will have value 255,
#     # denoting object boundaries. set to 0, denoting background class
#     pred[pred == 255] = 0
#     target[target == 255] = 0
#     # init empty array to hold IoU for each class
#     iou = np.zeros(n_classes)
#     # iterate through each class
#     for k in range(n_classes):
#         # calc the union of the pixels in pred and target labeled class k
#         union = np.count_nonzero(np.logical_or(pred == k, target == k))
#         union = union + 1e-10
#         # calc the intersection of the pixels in pred and target labeled class k
#         intersect = np.count_nonzero(np.logical_and(pred == k, target == k))
#         # calc the IoU for class k
#         iou[k] = intersect / union
#     # return the mean IoU of the pattern
#     return np.average(iou)
def iou_per_image(pred, target):
    ious = []
    for k in target.unique():
        union = torch.count_nonzero(torch.logical_or(pred == k, target == k))
        union = union + 1e-10

        intersect = torch.count_nonzero(torch.logical_and(pred == k, target == k))

        iou = intersect / union
        ious.append(iou)
    res = sum(ious) / len(target.unique())
    return float(res)



def iou(pred, target, n_classes: int = 21) -> float:
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
    iou = 0
    # iterate through each class
    for image in range(pred.shape[0]):
        iou  += iou_per_image(pred[image], target[image])
    iou = iou / pred.shape[0]
    return np.asarray(iou)

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


def plots(trainEpochLoss, valEpochLoss, valEpochAccuracy, valIoU, earlyStop, type="", saveLocation="test_"):
    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. earlyStop=-1 means the last epoch was the best one
    """
    # slice array according to early stop
    if earlyStop != -1:
        trainEpochLoss = trainEpochLoss[0:earlyStop + 1]
        valEpochLoss = valEpochLoss[0:earlyStop + 1]
        valEpochAccuracy = valEpochAccuracy[0:earlyStop + 1]
    # plotting
    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1, len(trainEpochLoss) + 1, 1)
    ax1.plot(epochs, trainEpochLoss, 'r', label=f'Training Loss')
    ax1.plot(epochs, valEpochLoss, 'g', label=f'Validation Loss')
    ax1.scatter(epochs[earlyStop], valEpochLoss[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax1.set_title(f'{type} Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(saveLocation + "loss.png")

    pd.DataFrame(trainEpochLoss).to_csv(saveLocation + "trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(saveLocation + "valEpochLoss.csv")
    pd.DataFrame(valIoU).to_csv(saveLocation + "valIoU.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(saveLocation + "valEpochAccuracy.csv")


def plot_predictions(image, mask, pred, save_location, i):
    """
    Helper function for plotting the predictions
    """
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].imshow(image.permute(1, 2, 0))
    ax[0].set_title("Image")
    ax[1].imshow(mask)
    ax[1].set_title("Mask")
    ax[2].imshow(pred)
    ax[2].set_title("Prediction")
    plt.savefig(save_location + "predictions" + str(i) + ".png")
