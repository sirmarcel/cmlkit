import numpy as np
import time
import logging
from datasets.autoload import load
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials
import qmmltools.inout as qmtio
import qmmltools.indices as qmti
import qmmltools.regression as qmtr
from qmmltools.model import ModelSpec
from qmmltools.mbtr.cached_mbtr import DiskAndMemCachedMBTR
from qmmltools.regression import loss, train_and_predict


def run_parallel(d, db):
    template = d['template']
    template['name'] = d['name']
    template['desc'] = 'Model during autotune run'

    if 'property' in d['data']:
        prop = d['data']['property']
    else:
        prop = d['data']['template']['property']

    objective_args = {'data': d['data'],
                      'opt': d['opt'],
                      'spec': d['template'],
                      'seeds': d['seeds']}

    trials = MongoTrials(db, exp_key=d['name'])

    logging.info("Setup done, please start some workers...")

    trials, duration = run_hypertune(objective_args,
                                     d['opt']['n_calls'],
                                     objective_with_loading, trials)

    to_save = {
        'db': db,
        'final_loss': trials.best_trial['result']['loss'],
        'final_loss_variance': trials.best_trial['result']['loss_variance'],
        'duration': duration,
        'post': {}
    }

    qmtio.save('out/' + d['name'] + '.run', to_save)

    top_3 = (np.array(trials.losses())).argsort()[:3]

    for i in range(3):
        spec = ModelSpec.from_dict(trials.trials[top_3[i]]['result']['spec_dict'])
        spec.name += '-' + 'best' + str(i)
        spec.save('out/')

    for i in range(3):
        spec = ModelSpec.from_dict(trials.trials[top_3[i]]['result']['spec_dict'])
        spec.name += '-' + 'best' + str(i)

        # The following is extremely ugly
        if 'post' in d.keys():
            for post_config in d['post']:

                post_data_train = load(post_config['data_train']['id'])
                post_data_test = load(post_config['data_test']['id'])

                if post_data_train.parent_info['id'] == post_data_test.parent_info['id']:
                    parent = load(post_data_train.parent_info['id'])
                    idx_train = post_data_train.idx
                    idx_test = post_data_test.idx

                    rep = DiskAndMemCachedMBTR(parent, spec, name=spec.name)
                    rep.save('out/')

                    l = qmtr.loss(parent, spec, rep, idx_train, idx_test, lossf=post_config['loss'], target_property=d['data']['property'])

                    info = "Training %i model on %s and predicting %s gave loss %s" % (i, post_config['data_train']['id'], post_config['data_test']['id'], str(l))

                    logging.info(info)

                    key = 'model' + str(i) + '_' + post_config['data_train']['id'] + '->' + post_config['data_test']['id']

                    to_save['post'][key] = l


    qmtio.save('out/' + d['name'] + '.run', to_save)


def objective_with_loading(args):
    np.random.seed(args['seeds']['seed_data'])

    data = load(args['data']['id'])

    splits = [qmti.threeway_split(data.n,
                                  args['data']['n_train'],
                                  args['data']['n_valid']) for i in range(args['opt']['n_cv'])]

    spec = ModelSpec.from_dict(args['spec'])
    logging.debug('Model MBTR settings' + str(spec.mbtrs))
    logging.debug('Model KRR settings' + str(spec.krr))

    rep = DiskAndMemCachedMBTR(data, spec)
    lossvec = []
    for rest, train, predict in splits:
        l = qmtr.loss(data, spec, rep, train, predict, args['opt']['loss'],
                      target_property=args['data']['property'])

        lossvec.append(l)

    mean_loss = np.mean(lossvec)
    var_loss = np.var(lossvec)

    logging.debug("Evaluated objective, loss was %.8f ± %.8f" % (mean_loss, var_loss))

    return {
        'loss': mean_loss,
        'loss_variance': var_loss,
        'status': STATUS_OK,
        'spec_dict': args['spec']
    }


def run_hypertune(objective_args, n_calls, objective, trials=Trials()):

    logging.info('Starting optimisation.')
    start = time.time()
    best = fmin(objective,
                space=objective_args,
                algo=tpe.suggest,
                max_evals=n_calls,
                trials=trials,
                verbose=1)
    end = time.time()
    duration = int(end - start)

    logging.info("Finished optimisation in %is, lowest achieved loss was %.8f ± %.8f" %
                 (duration, trials.best_trial['result']['loss'], trials.best_trial['result']['loss_variance']))

    return trials, duration
