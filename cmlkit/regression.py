import numpy as np
import warnings
import qmmlpack as qmml
import cmlkit.losses as cmls
from cmlkit import logger
from cmlkit.dataset import Dataset, Subset
from cmlkit.reps.mbtr import MBTR
from cmlkit.property_converter import convert


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


def train_model(data, spec, kernel_matrix=None, rep=None):
    """Train a kernel ridge regression model

    If a kernel_matrix is not given, attempt to compute it,
    computing the representation on the fly if necessary.

    Note that it is significantly faster to compute the rep
    (and possibly the kernel matrix) once, instead of doing
    it on the fly.

    Args:
        spec: ModelSpec
        data: Dataset or subclass
        kernel_matrix: optional, ndarray with the kernel matrix
        rep: optional, only used if no kernel matrix is given

    Returns:
        model: qmmlpack.KernelRidgeRegression instance
    """

    labels = data.p[spec.data['property']]

    if kernel_matrix is None and rep is not None:
        kernel_matrix = compute_kernel(spec.krr, rep.raw)
    elif kernel_matrix is None and rep is None:
        logger.debug('Training model without pre-computed rep and kernel_matrix, this might be slow.')
        kernel_matrix = compute_kernel(spec.krr, MBTR(data, spec).raw)

    return qmml.KernelRidgeRegression(kernel_matrix, labels, theta=(spec.krr['nl'],))


def train_and_predict(data_train, data_predict,
                      spec,
                      rep_train=None, rep_predict=None,
                      target_property=None,
                      return_intermediate=False):
    """Train a KRR model and predict some property

    This method expects different datasets for training and prediction,
    optionally with representations already computed. If a single dataset/
    representation is to be used with indices, use the idx_train_and_predict
    function instead.

    Args:
        data_train: Dataset for training
        data_predict: Dataset for prediction
        spec: ModelSpec
        rep_train: Optional, Representation of data_train
        rep_predict: Optional, Representation of data_test
        target_property: Optional, if set, predictions are converted to this property
        return_intermediate: Optional, if True, all intermediate data is returned as dict

    Returns:
        predictions: ndarray with predictions
        intermediate: dict with intermediate info (if requested)

    """
    if rep_train is None:
        logger.debug('Computing representation on the fly for {}.'.format(data_train.id))
        rep_train = MBTR(data_train, spec)
    else:
        sanity_check(data_train, spec, rep_train)

    if rep_predict is None:
        logger.debug('Computing representation on the fly for {}.'.format(data_predict.id))
        rep_predict = MBTR(data_predict, spec)
    else:
        sanity_check(data_predict, spec, rep_predict)

    kernel_train = compute_kernel(spec.krr, rep_train.raw)
    kernel_predict = compute_kernel(spec.krr, rep_train.raw, rep_predict.raw)

    model = train_model(data_train, spec, kernel_matrix=kernel_train)

    pred = model(kernel_predict)

    intermediate = {'kernel_train': kernel_train, 'kernel_predict': kernel_predict, 'model': model,
                    'rep_train': rep_train, 'rep_predict': rep_predict}

    if target_property is not None:
        intermediate['orig_pred'] = pred
        pred = convert(data_predict, pred, spec.data['property'], target_property)

    intermediate['pred'] = pred

    if return_intermediate is False:
        return pred
    else:
        return pred, intermediate


def compute_loss(data_train, data_predict,
                 spec,
                 rep_train=None, rep_predict=None,
                 target_property=None,
                 return_intermediate=False,
                 lossf=cmls.rmse):
    """Compute loss using explicit datasets and reps

    This function is intended for use with different instances of Dataset
    and representations, for instance for keeping training and test data
    separated. Use idx_compute_loss for the equivalent index-based function.

    Args:
        data_train: Dataset for training
        data_predict: Dataset for prediction
        spec: ModelSpec
        rep_train: Optional, Representation of data_train
        rep_predict: Optional, Representation of data_test
        target_property: Optional, if set, predictions are converted to this property
        return_intermediate: Optional, if True, all intermediate data is returned as dict
        lossf: If present, compute this loss (otherwise the rmse is computed)

    Returns:
        loss: Result of loss function, varying types
        intermediate: dict with intermediate info (if requested)

    """

    pred, intermediate = train_and_predict(data_train, data_predict,
                                           spec,
                                           rep_train, rep_predict,
                                           target_property,
                                           return_intermediate=True)

    if target_property is None:
        true = data_predict.p[spec.data['property']]
    else:
        intermediate['orig_true'] = data_predict.p[spec.data['property']]
        true = data_predict.p[target_property]

    intermediate['true'] = true

    intermediate['return_intermediate'] = return_intermediate
    loss = lossf(true, pred, context=intermediate)

    # some more complicated loss functions can optionally return
    # internal data (such as predictive variance et al), here we
    # capture that and splice it into the big return dict
    if isinstance(loss, dict):
        intermediate = {**intermediate, **loss}
        loss = loss['loss']

    if return_intermediate is False:
        return loss
    else:
        return loss, intermediate


def idx_train_and_predict(data, spec, idx_train, idx_predict, rep=None,
                          target_property=None, return_intermediate=False):
    """Train a KRR model and predict some property using indices

    This function is intended for use when the split of one dataset is investigated,
    i.e. there is a shared dataset (with optional rep) and indices specify what is the
    training and prediction set.

    Args:
        spec: ModelSpec
        data: Dataset or subclass
        idx_train: Index array for training
        idx_predict: Index array for prediction
        rep: Optional, Representation of data
        target_property: Optional, if set, predictions are converted to this property
        return_intermediate: Optional, if True, all intermediate data is returned as dict

    Returns:
        predictions: ndarray with predictions
        intermediate: dict with intermediate info (if requested)

    """
    if rep is None:
        logger.debug('Computing representation on the fly for {}.'.format(data.id))
        rep = MBTR(data, spec)
    else:
        sanity_check(data, spec, rep)

    return train_and_predict(data[idx_train], data[idx_predict],
                             spec,
                             rep_train=rep[idx_train], rep_predict=rep[idx_predict],
                             target_property=target_property, return_intermediate=return_intermediate)


def idx_compute_loss(data, spec, idx_train, idx_predict, rep=None,
                     target_property=None, return_intermediate=False,
                     lossf=cmls.rmse):
    """Compute loss using indices

    This function is intended for use when the split of one dataset is investigated,
    i.e. there is a shared dataset (with optional rep) and indices specify what is the
    training and prediction set.

    Args:
        data: Dataset
        spec: ModelSpec
        idx_train: Index array for training
        idx_predict: Index array for prediction
        rep: Optional, representation of Dataset
        lossf: If present, compute this loss (otherwise the rmse is computed)
        target_property: If present, predict this property
                         (otherwise default to the one specified by spec)
        return_intermediate: Optional, if True, all intermediate data is returned as dict

    Returns:
        loss: Float or list, depending on lossf
        intermediate: dict with intermediate info (if requested)

    """

    if rep is None:
        logger.debug('Computing representation on the fly for {}.'.format(data.id))
        rep = MBTR(data, spec)

    return compute_loss(data[idx_train], data[idx_predict],
                        spec, lossf=lossf,
                        rep_train=rep[idx_train], rep_predict=rep[idx_predict],
                        target_property=target_property, return_intermediate=return_intermediate)


def sanity_check(data, spec, rep):
    if isinstance(data, (Dataset, Subset)) and isinstance(rep, MBTR):
        if spec.hashes['mbtrs'] != rep.hashes['spec']:
            warnings.warn('The hash for the spec {} and the MBTR {} do not match!'.format(spec.name, rep.name))

        if data.hashes['geom'] != rep.hashes['geom']:
            warnings.warn('The hash for the geometries in dataset {} and the MBTR {} do not match!'.format(data.id, rep.name))
