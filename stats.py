import numpy as np


def pretty_loss(loss):
    """Return a string with a pretty-printed loss."""

    return "RMSE=%.5f  MAE=%.5f R2=%.5f" % loss


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


def logrmse(true, pred):
    """Compute log root mean squared error"""

    return rmse(np.log(true+1), np.log(pred+1))


def logmae(true, pred):
    """Compute log mean absolute error"""

    return mae(np.log(true+1), np.log(pred+1))


def logr2(true, pred):
    """Compute log Pearson's correlation coefficient"""

    return r2(np.log(true+1), np.log(pred+1))


def logloss(true, pred):
    """Return the triple (rmse, mae and r2)"""

    return (logrmse(true, pred), logmae(true, pred), logr2(true, pred) * 100)
