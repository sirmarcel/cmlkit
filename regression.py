import numpy as np
import qmmlpack as qmml
import qmmltools.stats as qmts
from property_converter import convert


def kernel(spec, rep, other_rep=None):
    """Compute the kernel matrix

    Args:
        spec: ModelSpec
        rep: Raw representation (i.e. ndarray)
        other_rep: If not None, compute the kernel between rep and other_rep

    Returns:
        kernel matrix, a ndarray
    """

    krr = spec.krr
    if krr['kernelf'][0] == 'gaussian':
        return qmml.kernel_gaussian(rep, other_rep, theta=(krr['kernelf'][1],))
    else:
        raise NotImplementedError('Unknown or no kernel found')


def train_model(data, spec, kernel_matrix):
    """Train a kernel ridge regression model

    Args:
        spec: ModelSpec
        data: Dataset or subclass
        kernel_matrix: ndarray

    Returns:
        model: qmmlpack.KernelRidgeRegression instance
    """

    labels = data.p[spec.data['property']]
    return qmml.KernelRidgeRegression(kernel_matrix, labels, theta=(spec.krr['nl'],))


def train_and_predict(data, spec, rep, train, predict, target_property=None):
    """Train a KRR model and predict some property

    Args:
        spec: ModelSpec
        data: Dataset or subclass
        rep: Representation of data
        train: Index array for training
        predict: Index array for prediction
        target_property: Optional, if set, predictions are converted to this property

    Returns:
        predictions: ndarray with predictions

    """

    kernel_matrix = kernel(spec, rep[train])
    model = train_model(data[train], spec, kernel_matrix)

    kernel_valid = kernel(spec, rep.raw[train], rep[predict])

    if target_property is None:
        return model(kernel_valid)
    else:
        pred = model(kernel_valid)
        return convert(data[predict], pred, spec.data['property'], target_property)


def loss(data, spec, rep, train, predict, lossf=qmts.mae, target_property=None):
    """Compute loss for given ModelSpec and representation"""

    prediction = train_and_predict(data, spec, rep, train, predict, target_property=target_property)

    if target_property is None:
        true = data[predict].p[spec.data['property']]
    else:
        true = data[predict].p[target_property]

    return lossf(true, prediction)
