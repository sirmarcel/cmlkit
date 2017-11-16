import numpy as np


def pretty_loss(loss):
    """Return a string with a pretty-printed loss."""

    return "RMSE=%.3f  MAE=%.3f R2=%.3f" % loss


def loss(true, pred):
    """Return the triple (rmse, mae and r2)"""

    return (rmse(true, pred), mae(true, pred), r2(true, pred) * 100)


def rmse(true, pred):
    "Compute root mean squared error"

    return np.sqrt(np.mean((pred - true)**2))


def mae(true, pred):
    "Compute mean absolute error"

    return np.mean(np.abs(pred - true))


def r2(true, pred):
    """Compute Pearson's correlation coefficient"""

    return np.corrcoef(true, pred)[0, 1]
