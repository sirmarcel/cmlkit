"""Implementations of various loss functions.

After some excursion into meta-programming magic,
I've decided to keep this as non-magic as possible.
Alas no getattr calls, and no decorators. KISS.

"""

import numpy as np


def rmse(true, pred, pv=None):
    """Root mean squared error."""
    return np.sqrt(np.mean((true - pred) ** 2))


def mae(true, pred, pv=None):
    """Mean absolute error."""
    return np.mean(np.fabs(true - pred))


def medianae(true, pred, pv=None):
    """Median absolute error."""
    return np.median(np.fabs(true - pred))


def maxae(true, pred, pv=None):
    """Maximum absolute error."""
    return np.max(np.fabs(true - pred))


def r2(true, pred, pv=None):
    """Squared correlation coefficient.

    Also known as square of product-moment correlation coefficient, Pearson's correlation.
    """
    return np.corrcoef(true, pred)[0, 1] ** 2


def one_minus_r2(true, pred, pv=None):
    """1 - R^2."""
    return 1.0 - r2(true, pred)


def mnlp(true, pred, pv):
    """Return the mean negative log probability.

    Only defined for models that predict a Gaussian
    probability distribution, like KRR or GPR."""

    raise NotImplementedError("Someone should implement mnlp loss.")


# Set some additional attributes by hand because decorators are tedious.

rmse.longname = "root_mean_squared_error"
rmse.needs_pv = False

mae.longname = "mean_absolute_error"
mae.needs_pv = False

medianae.longname = "median_absolute_error"
medianae.needs_pv = False

maxae.longname = "maximum_absolute_error"
maxae.needs_pv = False

r2.longname = "correlation_squared"
r2.needs_pv = False

one_minus_r2.longname = "one_minus_correlation_squared"
one_minus_r2.needs_pv = False

mnlp.longname = "mean_negative_log_probability"
mnlp.needs_pv = True

lossfs = {l.__name__: l for l in [rmse, mae, medianae, maxae, r2, one_minus_r2, mnlp]}


def get_lossf(name):
    """Obtain a callable for loss function name."""

    if callable(name):
        # we've been passed a function, let's do nothing
        return name
    elif name in lossfs:
        return lossfs[name]
    else:
        raise ValueError(f"Lossf or lossf group named {name} not found.")
