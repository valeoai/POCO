import numpy as np


def stats_overall_accuracy(cm):
    """Computes the overall accuracy.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.
    """
    return np.trace(cm) / cm.sum()


def stats_pfa_per_class(cm):
    """Computes the probability of false alarms.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.
    """
    sums = np.sum(cm, axis=0)
    mask = sums > 0
    sums[sums == 0] = 1
    pfa_per_class = (cm.sum(axis=0) - np.diag(cm)) / sums
    pfa_per_class[np.logical_not(mask)] = -1
    average_pfa = pfa_per_class[mask].mean()
    return average_pfa, pfa_per_class


def stats_accuracy_per_class(cm):
    """Computes the accuracy per class and average accuracy.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.

    # Returns
        average_accuracy: float.
            The average accuracy.
        accuracy_per_class: 1-D numpy array.
            The accuracy per class.
    """
    sums = np.sum(cm, axis=1)
    mask = sums > 0
    sums[sums == 0] = 1
    accuracy_per_class = np.diag(cm) / sums  # sum over lines
    accuracy_per_class[np.logical_not(mask)] = -1
    average_accuracy = accuracy_per_class[mask].mean()
    return average_accuracy, accuracy_per_class


def stats_iou_per_class(cm):
    """Computes the IoU per class and average IoU.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.

    # Returns
        average_accuracy: float.
            The average IoU.
        accuracy_per_class: 1-D numpy array.
            The IoU per class.
    """

    # compute TP, FN et FP
    TP = np.diagonal(cm, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(cm, axis=-1)
    TP_plus_FP = np.sum(cm, axis=-2)

    # compute IoU
    mask = TP_plus_FN == 0
    IoU = TP / (TP_plus_FN + TP_plus_FP - TP + mask)

    # replace IoU with 0 by the average IoU
    aIoU = IoU[np.logical_not(mask)].mean(axis=-1, keepdims=True)
    IoU += mask * aIoU

    return IoU.mean(axis=-1), IoU


def stats_f1score_per_class(cm):
    """Computes the F1 per class and average F1.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.

    # Returns
        average_accuracy: float.
            The average F1.
        accuracy_per_class: 1-D numpy array.
            The F1 per class.
    """
    # defined as 2 * recall * prec / recall + prec
    sums = np.sum(cm, axis=1) + np.sum(cm, axis=0)
    mask = sums > 0
    sums[sums == 0] = 1
    f1score_per_class = 2 * np.diag(cm) / sums
    f1score_per_class[np.logical_not(mask)] = -1
    average_f1_score = f1score_per_class[mask].mean()
    return average_f1_score, f1score_per_class
