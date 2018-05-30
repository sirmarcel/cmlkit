import numpy as np
import qmmlpack as qmml
import qmmltools.stats as qmts
from qmmltools.property_converter import convert


def compute_kernel(spec, rep, other_rep=None):
    """Compute the kernel matrix

    Supported kernels:
        - linear (no arguments)
        - gaussian (1 argument)
        - laplacian (1 argument)

    Args:
        spec: Dict specifying krr settings; format:
              {'kernelf': ('kernel_name', parameter)}
              or in the case of linear
              {'kernelf': 'linear'}
        rep: Raw representation (i.e. ndarray)
        other_rep: If not None, compute the kernel between rep and other_rep

    Returns:
        kernel matrix, a ndarray
    """

    if spec['kernelf'] == 'linear':
        return qmml.kernel_linear(rep, other_rep)
    elif spec['kernelf'][0] == 'gaussian':
        return qmml.kernel_gaussian(rep, other_rep, theta=(spec['kernelf'][1],))
    elif spec['kernelf'][0] == 'laplacian':
        return qmml.kernel_laplacian(rep, other_rep, theta=(spec['kernelf'][1],))
    else:
        raise NotImplementedError('Unknown kernel specification {}!'.format(spec['kernelf']))


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

    This function is intended for use when the split of one dataset are investigated.

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

    kernel_matrix = compute_kernel(spec.krr, rep.raw[train])
    model = train_model(data[train], spec, kernel_matrix)

    kernel_valid = compute_kernel(spec.krr, rep.raw[train], rep.raw[predict])

    if target_property is None:
        return model(kernel_valid)
    else:
        pred = model(kernel_valid)
        return convert(data[predict], pred, spec.data['property'], target_property)


def loss(data, spec, rep, train, predict, lossf=qmts.loss, target_property=None):
    """Compute loss for given ModelSpec and representation, using indices

    This function is intended for use when the split of one dataset are investigated,
    as opposed to the one below, where different datasets/reps are used for training
    and prediction.

    Args:
        data: Dataset
        spec: ModelSpec
        rep: Representation of Dataset
        train: Training indices
        predict: Prediction indices
        lossf: If present, compute this loss
        target_property: If present, predict this property 
                         (otherwise default to the one specified by spec)

    Returns:
        loss: Float or list, depending on lossf

    """

    prediction = train_and_predict(data, spec, rep, train, predict, target_property=target_property)

    if target_property is None:
        true = data[predict].p[spec.data['property']]
    else:
        true = data[predict].p[target_property]

    return lossf(true, prediction)



def train_and_predict_with_datasets_and_reps(data_train, data_predict,
                                             spec,
                                             rep_train, rep_predict,
                                             target_property=None):
    """Train a KRR model and predict some property, not using indices

    Args:
        data_train: Dataset for training
        data_predict: Dataset for prediction
        spec: ModelSpec
        rep_train: Representation of data_train
        rep_predict: Representation of data_test
        target_property: Optional, if set, predictions are converted to this property

    Returns:
        predictions: ndarray with predictions

    """

    kernel_matrix = compute_kernel(spec.krr, rep_train.raw)
    model = train_model(data_train, spec, kernel_matrix)

    kernel_valid = compute_kernel(spec.krr, rep_train.raw, rep_predict.raw)

    if target_property is None:
        return model(kernel_valid)
    else:
        pred = model(kernel_valid)
        return convert(data_predict, pred, spec.data['property'], target_property)


def loss_with_datasets_and_reps(data_train, data_predict,
                                spec,
                                rep_train, rep_predict,
                                target_property=None, lossf=qmts.loss):

    """Compute loss using explicit datasets and reps

    This function is intended for use with different instances of Dataset
    and representations, for instance for keeping training and test data
    separated.

    Args:
        data_train: Dataset for training
        data_predict: Dataset for prediction
        spec: ModelSpec
        rep_train: Representation of data_train
        rep_predict: Representation of data_test
        lossf: If present, compute this loss
        target_property: If present, predict this property 
                         (otherwise default to the one specified by spec)

    Returns:
        loss: Float or list, depending on lossf

    """

    prediction = train_and_predict_with_datasets_and_reps(data_train, data_predict,
                                                          spec,
                                                          rep_train, rep_predict,
                                                          target_property=target_property)

    if target_property is None:
        true = data_predict.p[spec.data['property']]
    else:
        true = data_predict.p[target_property]

    return lossf(true, prediction)

