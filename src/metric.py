from typing import Union
import numpy as np
import torch


def pfbeta(
    labels: np.ndarray,
    predictions: Union[np.ndarray, torch.Tensor],
    beta: int = 1
):
    """
    Calculates probabilistic F1-score on samples

    Args:
        labels: true labels
        predictions: predicted probabilities
        beta: hyperparameter bets for F1-score

    Output:
        pfbeta_f1score: probabilistic F1-score
    """
    y_true_count = 0
    ctp, cfp = 0, 0
    eps = 10**(-6)

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + eps)
    c_recall = ctp / (y_true_count + eps)
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + eps)
        return result[0].numpy() if type(result) is not int else result
    else:
        return 0
