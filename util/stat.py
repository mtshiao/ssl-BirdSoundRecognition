import numpy as np
from scipy import stats
from sklearn import metrics
import torch
from pathlib import Path

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def save_output_and_target(output, target, output_file='predictions.csv', target_file='targets.csv'):
    """
    Save the output (predicted probabilities) and target (one-hot encodings) to CSV files.

    Args:
      output: numpy array, predicted probabilities with shape (samples_num, classes_num)
      target: numpy array, one-hot encodings with shape (samples_num, classes_num)
      output_file: string, filename to save the predicted probabilities
      target_file: string, filename to save the one-hot encodings
    """
    # Ensure the output directory exists; if not, you might need to create it or modify the path accordingly.
    np.savetxt(output_file, output, delimiter=',', fmt='%f')
    np.savetxt(target_file, target, delimiter=',', fmt='%d')

def calculate_stats(output, target, evalbool):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    if evalbool:
        output_file = Path.cwd().parents[0].joinpath('report', 'model_new', 'testing', 'pre.csv')
        target_file = Path.cwd().parents[0].joinpath('report', 'model_new', 'testing', 'tar.csv')
        save_output_and_target(output, target, output_file, target_file)

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    return stats