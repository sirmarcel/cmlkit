import numpy as np
import logging
from hyperopt import STATUS_OK
from qmmltools.model_spec import ModelSpec
import qmmltools.autoload
from qmmltools.mbtr.cached_mbtr import DiskAndMemCachedMBTR
import qmmltools.regression as qmtr
import qmmltools.indices as qmti


def objective(d):
    spec = d['spec']
    data = d['data']
    config = d['config']
    internal = d['internal']

    spec = ModelSpec.from_dict(spec)
    logging.debug('Model MBTR settings' + str(spec.mbtrs))
    logging.debug('Model KRR settings' + str(spec.krr))

    dataset = qmmltools.autoload.dataset(data['id'])

    rep = DiskAndMemCachedMBTR(dataset, spec)
    logging.debug('Successfully computed the MBTR!')

    splits = generate_cv_idx(config['cv'], dataset.n)

    lossvec = []
    for rest, train, test in splits:
        loss = qmtr.loss(dataset, spec, rep, train, test, config['loss'],
                         target_property=data['property'])

        lossvec.append(loss)

    mean_loss = np.mean(lossvec)
    var_loss = np.var(lossvec)

    logging.debug("Evaluated objective, loss was %.8f ± %.8f" % (mean_loss, var_loss))

    return {
        'loss': mean_loss,
        'loss_variance': var_loss,
        'status': STATUS_OK,
        'spec_dict': d['spec']
    }


def generate_cv_idx(cv_config, n):

    if cv_config['type'] == 'random':
        splits = [qmti.threeway_split(n,
                                      cv_config['n_train'],
                                      cv_config['n_valid']) for i in range(cv_config['n_cv'])]

    else:
        raise NotImplementedError("Currently only random CV is supported.")

    return splits
