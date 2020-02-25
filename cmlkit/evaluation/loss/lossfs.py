"""Implementations of various loss functions.

After some excursion into meta-programming magic,
I've decided to keep this as non-magic as possible.
Alas no getattr calls, and no decorators. KISS.

"""

import numpy as np


def rmse(true, pred, pv=None):
    """Root mean squared error."""
    return np.sqrt(np.mean((true - pred) ** 2))


def rmsle(true, pred, pv=None):
    """Root mean squared log error.

    Measure used in the Nomad 2018 kaggle competition,
    useful to compare losses for quantities with differing
    orders of magnitude.
    """
    return np.sqrt(np.mean(np.log((pred + 1.0) / (true + 1.0)) ** 2))


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
    """Squared Pearson product-moment correlation coefficient.

    Informally speaking, this measures how well true and predicted
    values correlate in a linear way, with 0 being "not at all"
    and 1 being "perfectly". Note that this makes no statement
    about the *sign* of the correlation, i.e. negative correlation
    also counts. Looking at a plot and other errors metrics is advised!

    WARNING: This is *not* identical to the coefficient of
    determination, but used synonymously in some cases. This is
    due to the fact that for some models (linear least-squares
    multiple regression) the definitions can be shown to be equivalent.

    For KRR, this r2 is typically used instead of the more general definition.
    """
    return np.corrcoef(true, pred)[0, 1] ** 2


def cod(true, pred, pv=None):
    """Coefficient of determination.

    Also often termed R2 or r2. See docstring
    of `r2` loss.

    Roughly speaking, the squared error scaled
    by the variance of the data. 1.0 is best.

    Can be negative, but <= 1.0.

    """

    mean = np.mean(true)
    sum_of_squares = np.sum((true - mean)**2)
    sum_of_residuals = np.sum((true - pred)**2)

    return 1.0 - (sum_of_residuals/sum_of_squares)


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

rmsle.longname = "root_mean_squared_log_error"
rmsle.needs_pv = False

mae.longname = "mean_absolute_error"
mae.needs_pv = False

medianae.longname = "median_absolute_error"
medianae.needs_pv = False

maxae.longname = "maximum_absolute_error"
maxae.needs_pv = False

r2.longname = "pearson_correlation_squared"
r2.needs_pv = False

cod.longname = "coefficient_of_determination"
cod.needs_pv = False

one_minus_r2.longname = "one_minus_pearson_correlation_squared"
one_minus_r2.needs_pv = False

mnlp.longname = "mean_negative_log_probability"
mnlp.needs_pv = True

lossfs = {
    l.__name__: l for l in [rmse, rmsle, mae, medianae, maxae, r2, cod, one_minus_r2, mnlp]
}


def get_lossf(name):
    """Obtain a callable for loss function name."""

    if callable(name):
        # we've been passed a function, let's do nothing
        return name
    elif name in lossfs:
        return lossfs[name]
    else:
        raise ValueError(f"Lossf or lossf group named {name} not found.")
