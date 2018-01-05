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

    # TODO

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
