import numpy as np
from hyperopt import STATUS_OK, STATUS_FAIL
import qmmlpack as qmml
import time
from cmlkit import logger
from cmlkit.model_spec import ModelSpec
from cmlkit.autoload import load_dataset_cached
from cmlkit.reps.cached_mbtr import DiskAndMemCachedMBTR
import cmlkit.regression as cmlr
import cmlkit.indices as cmli
import cmlkit.helpers as cmlh
import cmlkit.autotune.local_grid_search as lgs


def objective(d):
    cmlh.tuples_to_lists(d)
    locs = cmlh.find_pattern(d, lgs.lgs_pattern)

    if len(locs) > 0:
        # lgs expects a real-valued objective, so we must wrap it
        def lgs_objective(template):
            return _objective(d)['loss']

        try:

            lgs.run_lgs(d, lgs_objective,
                        resolution=d['config']['lgs']['resolution'], maxevals=d['config']['lgs']['maxevals'],
                        ignore=[['config'], ['internal']])

        except qmml.QMMLException as e:
            logger.error('Encountered QMMLException during local grid search. Will mark this trial as failed. Error text below.')
            logger.error(e)
            return {
                'status': STATUS_FAIL,
                'spec_dict': d['spec'],
                'loss': float('inf')
            }

        return _objective(d)
    else:
        return _objective(d)


def _objective(d):
    start = time.time()
    spec = d['spec']
    data = d['data']
    config = d['config']
    # internal = d['internal']

    spec = ModelSpec.from_dict(spec)
    logger.debug('Model MBTR settings' + str(spec.mbtrs))
    logger.debug('Model KRR settings' + str(spec.krr))

    try:
        if config['cv'] is not None:
            mean_loss, var_loss = compute_loss_with_cv(data, spec, config)
        elif 'id-train' and 'id-test' in data:
            if 'id' in data:
                mean_loss, var_loss = compute_loss_with_idx(data, spec, config)
            else:
                mean_loss, var_loss = compute_loss(data, spec, config)
        else:
            msg = 'Could not find either cross-validation instructions or train/validation sets. Aborting.'
            logger.error(msg)
            raise(msg)

        end = time.time()
        duration = end - start
        logger.debug("Evaluated objective in {}s, loss was {} ± {}".format(duration, mean_loss, var_loss))

        return {
            'loss': mean_loss,
            'loss_variance': var_loss,
            'status': STATUS_OK,
            'spec_dict': d['spec']
        }

    except qmml.QMMLException as e:
        logger.error('Encountered QMMLException during objective evaluations. Will mark this trial as failed. Error text below.')
        logger.error(e)

        return {
            'status': STATUS_FAIL,
            'spec_dict': d['spec'],
            'loss': float('inf')
        }


def compute_loss_with_cv(data, spec, config):

    dataset = load_dataset_cached(data['id'])

    rep = DiskAndMemCachedMBTR(dataset, spec)
    logger.debug('Successfully computed the MBTR, moving on to predictions.')

    splits = generate_cv_idx(config['cv'], dataset.n)

    lossvec = []

    for rest, train, test in splits:
        loss = cmlr.idx_compute_loss(dataset, spec, train, test, rep=rep, lossf=config['loss'], target_property=data['property'])

        lossvec.append(loss)

    mean_loss = np.mean(lossvec)
    var_loss = np.var(lossvec)

    return mean_loss, var_loss


def generate_cv_idx(cv_config, n):

    if cv_config[0] == 'random':
        splits = [cmli.threeway_split(n,
                                      cv_config[2],
                                      cv_config[3]) for i in range(cv_config[1])]

    else:
        raise NotImplementedError("Currently only random CV is supported.")

    return splits


def compute_loss(data, spec, config):
    data_train = load_dataset_cached(data['id-train'])
    data_test = load_dataset_cached(data['id-test'])

    rep_train = DiskAndMemCachedMBTR(data_train, spec)
    logger.debug('Successfully computed the training MBTR.')
    rep_test = DiskAndMemCachedMBTR(data_test, spec)
    logger.debug('Successfully computed the testing MBTR.')

    loss = cmlr.compute_loss(data_train, data_test,
                             spec,
                             rep_train=rep_train, rep_predict=rep_test,
                             target_property=data['property'],
                             lossf=config['loss'])

    return loss, 0


def compute_loss_with_idx(data, spec, config):
    data_train = load_dataset_cached(data['id-train'])
    data_test = load_dataset_cached(data['id-test'])
    data_master = load_dataset_cached(data['id'])

    rep = DiskAndMemCachedMBTR(data_master, spec)
    logger.debug('Successfully computed the MBTR.')

    loss = cmlr.idx_compute_loss(data_master, spec,
                                 data_train.idx, data_test.idx,
                                 rep=rep,
                                 target_property=data['property'],
                                 lossf=config['loss'])

    return loss, 0
