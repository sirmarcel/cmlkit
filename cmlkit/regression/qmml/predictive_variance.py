"""Computing the predictive variance.

Note: This is currently not used anywhere, and is simply
      being kept for future use.

      The rest of this module needs to first support
      - Some way to define whether a kernel is normalised.
        (Plus the logic to compute the additional kernel if not.)
        (Plus deciding whether we need the covariance at any point.)
      - A considered way to modify the predict method of KRR.

***

While cmlkit.regression.qmml is rooted in the KRR view of things, it is
very useful to occasionally adopt the GPR view and compute
predictive variances as measure of uncertainty of a given
prediction.

Our reference for this is "Gaussian Processes for Machine Learning"
by Rasmussen and Williams, 2006, ISBN 978-262-18253-9.

Eq. 2.24 (page 16) gives the main equation we're concerned with:

cov(f*) = K(X*, X*) - K(X*, X) [K(X, X) + nl*I]^-1 K(X, X*)

Where the * marks the points at which we're trying to predict.
nl = sigma^2 is the regularisation parameter, or noise level.

Note that unless the kernel is normalised so a point evaluated with
itself has kernel distance 1, we also need the kernel between
all points we're predicting.

This gives a straightfoward way to compute the full covariance for
an entire set of predictions. Unfortunately, the matrix inversion
in this equation is unstable, so we need to use the Cholesky decompostion
instead.

***

Dropping to the single-point case for second, we let k* be a vector with the
kernel evaluations between all training points and the one point at
which we're predicting, and k** the kernel with itself.

    pv(x*) = k** - k*^T . (K + 1*nl)^-1 . k*

Let L be the lower triangular Cholesky decomposition. Then the inverse of the
augmented kernel matrix is

    (K + 1*nl)^-1 = L^-1,T L^-1

Which we can multiply with the k*, and then the inversion of L becomes solving
the problem

    L v = k,

which can be done by forward substitution. (L is triangular.) Then the PV is just

    pv(x*) = k** - v^T.v

***

This extends straightforwardly to the "vectorised" case where we're dealing with
many points at once. The only different is that the final result is the full
covariance matrix, the diagonal of which is the predictive variance.

Note also that this is reasonably cheap to compute, since the Cholesky decomposition
is already performed to train the model. (However, computing kstar might not be cheap.)

"""

import qmmlpack


def predictive_variance(model, kernel_matrix, kstar=1):
    """Compute the predictive variance.

    Note that if centering is enabled for the KRR model, the kernel_matrix
    needs to be centered as well, which this functions does automatically.

    Args:
        model: A qmmlpack.regression.KernelRidgeRegression instance
        kernel_matrix: ndarray with each column representing kernel
                       evaluations between a new point and all training points
        kstar: Either 1, in which instance a kernel in which points
               evaluated with themselves result in 1, otherwise a 1-d ndarray
               with kernel evaluations of each new point with itself

    Returns:
        ndarray with the predictive covariance for all columns of kernel_matrix
    """

    if model._centering:
        L = qmmlpack.center_kernel_matrix(model.kernel_matrix, l=kernel_matrix)
        v = qmmlpack.forward_substitution(model.cholesky_decomposition.T, L)
    else:
        v = qmmlpack.forward_substitution(model.cholesky_decomposition.T, kernel_matrix)

    pv = kstar - np.diag(v.T @ v)

    return pv
