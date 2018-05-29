import numpy as np
import logging
import time
import qmmltools.inout as qmtio
import qmmltools.autoload as qmta
from hyperopt import fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials
from qmmltools.autotune.objective import objective
from qmmltools.autotune.parse import preprocess
from qmmltools.model_spec import ModelSpec


def run(task):
    if isinstance(task, str):
        task = qmtio.read_yaml(task)

    preprocess(task)

    if task['config']['parallel'] is True:
        trials = MongoTrials(task['config']['db'], exp_key=task['name'])
        logging.info('Performing parallel run. Remember to start workers!')
    else:
        trials = Trials()
        logging.info('Performing serial run.')

    trials, duration = run_hyperopt(task, trials)

    postprocess(task, trials, duration)


def postprocess(task, trials, duration):
    to_save = {
        'final_loss': trials.best_trial['result']['loss'],
        'final_loss_variance': trials.best_trial['result']['loss_variance'],
        'duration': duration,
        'post': {}
    }

    qmtio.save('out/' + task['name'] + '.run', to_save)

    top = (np.array(trials.losses())).argsort()[:task['config']['n_cands']]

    for i in range(task['config']['n_cands']):
        spec = ModelSpec.from_dict(trials.trials[top[i]]['result']['spec_dict'])
        spec.name += '-' + 'best' + str(i)
        spec.save('out/')

    logging.info('Saved result ModelSpecs, moving on to additional tasks.')


def run_hyperopt(task, trials):

    logging.info('Starting optimisation.')
    start = time.time()
    best = fmin(objective,
                space=task,
                algo=tpe.suggest,
                max_evals=task['config']['n_calls'],
                trials=trials,
                verbose=1)
    end = time.time()
    duration = int(end - start)

    logging.info("Finished optimisation in %is, lowest achieved loss was %.8f ± %.8f" %
                 (duration, trials.best_trial['result']['loss'], trials.best_trial['result']['loss_variance']))

    return trials, duration
