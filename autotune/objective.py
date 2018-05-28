import numpy as np
import logging
from qmmltools.model_spec import ModelSpec
import qmmltools.autoload
from qmmltools.mbtr.cached_mbtr import DiskAndMemCachedMBTR
import qmmltools.regression as qmtr


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

    splits = internal['splits']

    lossvec = []
    for rest, train, test in splits:
        loss = qmtr.loss(dataset, spec, rep, train, predict, config['loss'],
                         target_property=data['property'])

        lossvec.append(loss)

    mean_loss = np.mean(lossvec)
    var_loss = np.var(lossvec)

    logging.debug("Evaluated objective, loss was %.8f ± %.8f" % (mean_loss, var_loss))

    return {
        'loss': mean_loss,
        'loss_variance': var_loss,
        'status': STATUS_OK,
        'spec_dict': args['spec']
    }
