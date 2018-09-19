import numpy as np
import cmlkit.predictive_variance as cmlpv


def pretty_loss(loss):
    """Return a string with a pretty-printed loss."""

    return "RMSE=%.5f  MAE=%.5f R2=%.5f" % loss


def loss(true, pred, context=None):
    """Return the triple (rmse, mae and r2)"""

    return (rmse(true, pred), mae(true, pred), r2(true, pred))


def rmse(true, pred, context=None):
    "Compute root mean squared error"

    return np.sqrt(np.mean((pred - true)**2))


def mae(true, pred, context=None):
    "Compute mean absolute error"

    return np.mean(np.abs(pred - true))


def r2(true, pred, context=None):
    """Compute Pearson's correlation coefficient"""

    return np.corrcoef(true, pred)[0, 1]


def logrmse(true, pred, context=None):
    """Compute log root mean squared error"""

    return rmse(np.log(true + 1), np.log(pred + 1))


def logmae(true, pred, context=None):
    """Compute log mean absolute error"""

    return mae(np.log(true + 1), np.log(pred + 1))


def logr2(true, pred, context=None):
    """Compute log Pearson's correlation coefficient"""

    return r2(np.log(true + 1), np.log(pred + 1))


def logloss(true, pred, context=None):
    """Return the triple (rmse, mae and r2)"""

    return (logrmse(true, pred), logmae(true, pred), logr2(true, pred))


def mnll(true, pred, context={}):
    """Return the mean negative log likelihood.

    Only defined for models that predict a Gaussian
    probability distribution, like KRR or GPR.

    Args:
        true: True labels
        pred: Predicted labels (predictive mean)
        context: Dict containing the keys 'model' and 'kernel_predict',
                 which will then be forwarded to cmlkit.predictive_variance
                 to compute the variance on the fly.
                 If the key 'return_intermediate' is True, the variances
                 and full nll are returned in a dict.

    """
    kstar = context.get('kstar', 1.0)

    pv = cmlpv.predictive_variance(model=context['model'], kernel_matrix=context['kernel_predict'], kstar=kstar)
    nll = cmlpv.negative_log_likelihood(true, pred, var=pv)

    mnll = np.mean(nll)

    if context.get('return_intermediate', False):
        return {'loss': mnll, 'pv': pv, 'nll': nll}
    else:
        return mnll


def smnll(true, pred, context={}):
    """Return the standardised mean negative log likelihood.

    Same as mnll, but offset by the mnll of a model that predicts
    training mean and variance.

    Args:
        true: True labels
        pred: Predicted labels (predictive mean)
        context: Dict containing the keys 'model' and 'kernel_predict',
                 which will then be forwarded to cmlkit.predictive_variance
                 to compute the variance on the fly.
                 If the key 'return_intermediate' is True, the variances
                 and full nll are returned in a dict.

    """
    pv = cmlpv.predictive_variance(model=context['model'], kernel_matrix=context['kernel_predict'])
    nll = cmlpv.negative_log_likelihood(true, pred, var=pv)

    mnll = np.mean(nll)

    labels = context['model'].labels
    norm_var = np.zeros_like(true) + np.var(labels)
    norm_mean = np.zeros_like(true) + np.mean(labels)

    offset = np.mean(cmlpv.negative_log_likelihood(true, norm_mean, var=norm_var))

    mnll -= offset

    if context.get('return_intermediate', False):
        return {'loss': mnll, 'pv': pv, 'nll': nll}
    else:
        return mnll


def loonll(true, pred, context={}):
    """Return the mean leave-one-out negative log likelihood.

    This is a bit of a hack, since this doesn't depend on the
    true and predicted labels at all, but rather is a retrospective
    measure on the training set.

    Args:
        true: True labels
        pred: Predicted labels (predictive mean)
        context: Dict containing the key 'model'.

    """
    nll, mu, sigma = cmlpv.loo_negative_log_likelihood(context['model'])

    mnll = np.mean(nll)

    if context.get('return_intermediate', False):
        return {'loss': mnll, 'pv': sigma, 'nll': nll, 'mu': mu}
    else:
        return mnll
