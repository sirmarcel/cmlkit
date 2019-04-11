import numpy as np
import qmmlpack.validation as val
import cmlkit.predictive_variance as cmlpv


# loss functions
def rmse(true, pred, pv=None):
    """Root mean squared error."""
    return np.sqrt(np.mean((true - pred)**2))


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
    return np.corrcoef(true, pred)[0, 1]


def one_minus_r2(true, pred, pv=None):
    """1 - R^2."""
    return 1. - r2(true, pred)


# Losses with predictive variance

def mnlp(true, pred, pv):
    """Return the mean negative log probability.

    Only defined for models that predict a Gaussian
    probability distribution, like KRR or GPR."""

    nll = cmlpv.negative_log_likelihood(true, pred, var=pv)

    mnll = np.mean(nll)

    return mnll

# We do this because decorators obscure the function name
# when debugging/getting stack traces


rmse.name = rmse.__name__
rmse.longname = 'root_mean_squared_error'
rmse.needs_pv = False

mae.name = mae.__name__
mae.longname = 'mean_absolute_error'
mae.needs_pv = False

medianae.name = medianae.__name__
medianae.longname = 'median_absolute_error'
medianae.needs_pv = False

maxae.name = maxae.__name__
maxae.longname = 'maximum_absolute_error'
maxae.needs_pv = False

r2.name = r2.__name__
r2.longname = 'correlation_squared'
r2.needs_pv = False

one_minus_r2.name = one_minus_r2.__name__
one_minus_r2.longname = 'one_minus_correlation_squared'
one_minus_r2.needs_pv = False

mnlp.name = mnlp.__name__
mnlp.longname = 'Mean negative log probability'
mnlp.needs_pv = True


# some data structures of losses

default_losses = [rmse, mae, r2]
pv_losses = [mnlp]
pred_losses = [rmse, mae, medianae, maxae, r2, one_minus_r2]
all_losses = [*pv_losses, *pred_losses]

groups = {
    'default': [rmse, mae, r2],
    'pv': [mnlp],
    'pred': [rmse, mae, medianae, maxae, r2, one_minus_r2],
    'all': [rmse, mae, medianae, maxae, r2, one_minus_r2, mnlp]
}
