import qmmlpack as qmml
import numpy as np


def predictive_variance(model, kernel_matrix, kstar=1):
    """Compute the predictive variance of the model for the points in the kernel matrix.

    The kernel matrix is a matrix of kernel evaluations between training
    points and new points to predict. Every column therefore represents a
    single 'new' point.


    For one such column, which we call k*, the predictive variance is defined
    as:

    k** - k*^T . (K + 1*nl)^-1 . k*

    K is the training kernel matrix and nl the regularisation parameter.
    k** is the evaluation of the training point with itself.

    The implementation here avoids the explicit inversion of the augmented
    Kernel matrix (which is numerically unstable and slow) by using the
    Cholesky decomposition, using a formula given in 'Gaussian Processes' by
    Rasmussen & Williams:

    Let L be the lower triangular Cholesky decomposition. Then the inverse of the
    augmented kernel matrix is

    (K + 1*nl)^-1 = L^-1,T L^-1

    Which we can multiply with the k*, and then the inversion of L becomes solving
    the problem

    L v = k,

    which can be done by forward substitution. (L is triangular.) Then the PV is just

    PV = v^T.v ! Easy.

    Note that if centering is enabled for the regression model, the kernel_matrix
    needs to be centered as well.

    Args:
        model: A qmmlpack.regression.KernelRidgeRegression instance
        kernel_matrix: ndarray with each column representing kernel
                       evaluations between a new point and all training points
        kstar: Either 1, in which instance a kernel in which points
               evaluated with themselves result in 1, otherwise a 1-d ndarray
               with kernel evaluations of each new point with itself

    Returns:
        pv: 1-d ndarray with the predictive variance for all columns of kernel_matrix
    """

    if model._centering:
        L = qmml.kernels_centering.center_kernel_matrix(model.kernel_matrix, l=kernel_matrix)
        v = qmml.numerics.forward_substitution(model.cholesky_decomposition.T, L)
    else:
        v = qmml.numerics.forward_substitution(model.cholesky_decomposition.T, kernel_matrix)

    pv = kstar - np.diag(v.T @ v)

    return pv


def loo_predictive_variance(model):
    """Compute the log leave-one-out predictive variance.

    As defined in eq. 5.10 of Rasmussen and Williams:

    \log PV = - \frac{1}{2} \log \sigma_i^2 
              - \frac{(y - \mu_i)^2}{2 \sigma_i^2} - \frac{1}{2} \log{2\pi}

    \sigma_i^2 = 1/K^{-1}_{ii} \\
    \mu_i = y_i - (K^{-1} y)/(K^{-1}_{ii})

    (K is the augmented kernel matrix)

    This is the log of the predictive variance, for a model
    that was trained on all data, but without that particular point.

    It is therefore a retrospective measure!

    Args:
        model: qmmlpack.regression.KernelRidgeRegression instance

    Returns:
        pv: 1-d ndarray with the log leave-one-out predictive variance for all training points
        y: Labels centred to their mean
        mu: Predictive mean (leaving out the ith training point)

    """

    u = model.cholesky_decomposition
    y = model.labels

    if model.centering is False:
        y -= np.mean(y)

    uinv = np.linalg.inv(u)
    kinv = uinv @ np.transpose(uinv)

    sigma = 1 / np.diag(kinv)
    mu = y - (kinv @ y)/np.diag(kinv)

    pv = -0.5*np.log(sigma) - (y - mu)**2/(2*sigma) - 0.5*np.log(2*np.pi)

    return pv, y, mu
