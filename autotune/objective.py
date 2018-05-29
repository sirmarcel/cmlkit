import numpy as np
from hyperopt import STATUS_OK
from qmmltools import logger
from qmmltools.model_spec import ModelSpec
from qmmltools.autoload import load_dataset
from qmmltools.mbtr.cached_mbtr import DiskAndMemCachedMBTR
import qmmltools.regression as qmtr
import qmmltools.indices as qmti


def objective(d):
    spec = d['spec']
    data = d['data']
    config = d['config']
    internal = d['internal']

    spec = ModelSpec.from_dict(spec)
    logger.debug('Model MBTR settings' + str(spec.mbtrs))
    logger.debug('Model KRR settings' + str(spec.krr))

    dataset = load_dataset(data['id'])

    rep = DiskAndMemCachedMBTR(dataset, spec)
    logger.debug('Successfully computed the MBTR!')

    splits = generate_cv_idx(config['cv'], dataset.n)

    lossvec = []
    for rest, train, test in splits:
        loss = qmtr.loss(dataset, spec, rep, train, test, config['loss'],
                         target_property=data['property'])

        lossvec.append(loss)

    mean_loss = np.mean(lossvec)
    var_loss = np.var(lossvec)

    logger.debug("Evaluated objective, loss was %.8f ± %.8f" % (mean_loss, var_loss))

    return {
        'loss': mean_loss,
        'loss_variance': var_loss,
        'status': STATUS_OK,
        'spec_dict': d['spec']
    }


def generate_cv_idx(cv_config, n):

    if cv_config[0] == 'random':
        splits = [qmti.threeway_split(n,
                                      cv_config[2],
                                      cv_config[3]) for i in range(cv_config[1])]

    else:
        raise NotImplementedError("Currently only random CV is supported.")

    return splits
