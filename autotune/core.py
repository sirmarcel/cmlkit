import qmmltools.indices as qmti
import qmmltools.inout as qmtio
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials
from qmmltools.autotune.objective import objective
from qmmltools.autotune.settings import parse_settings


def run(task):
    if isinstance(task, str):
        task = qmtio.read_yaml(task_file)

    # Pre-process task settings
    parse_settings(task)

    splits = generate_cv_idx(task['config']['cv'])

    if task['config']['parallel'] = True:
        trials = MongoTrials(task['config']['db'], exp_key=task['name'])
        logging.info('Performing parallel run. Remember to start workers!')
    else:
        trials = Trials()
        logging.info('Performing serial run.')

    trials, duration = run_hypertune(task, task['config']['n_calls'], trials)




def post(trials, duration, task):
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



def generate_cv_idx(cv_config, n):

    if cv['type'] == 'random':
        splits = [qmti.threeway_split(n,
                                      cv_config['n_train'],
                                      cv_config['n_valid']) for i in range(cv_config['n_cv'])]

    else:
        raise NotImplementedError("Currently only random CV is supported.")

    return splits


def run_hyperopt(space, n_calls, trials):

    logging.info('Starting optimisation.')
    start = time.time()
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=n_calls,
                trials=trials,
                verbose=1)
    end = time.time()
    duration = int(end - start)

    logging.info("Finished optimisation in %is, lowest achieved loss was %.8f ± %.8f" %
                 (duration, trials.best_trial['result']['loss'], trials.best_trial['result']['loss_variance']))

    return trials, duration
