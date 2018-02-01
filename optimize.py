from skopt import gp_minimize
import logging
import time


def optimize_bayes(f, params, n_calls=25, acq_func='EI', n_random_starts=5, random_state=2342):
    """Optimize f with Bayesian optimisation, for the parameters supplied.

    Uses Bayesian optimisation to minimise the output of f, which is a
    function of n parameters (and nothing else).

    For details of the optimisation method, see the skopt docs.

    Args:
        f: A callable; for example f(a, b, c)
        params: A dict, containing the label for the parameters and their bounds,
                params = {'a': (0, 1), 'b': (2, 3), 'c': (5, 6)}. The labels must
                have the same name as the function arguments!
        n_calls: Number of calls made to f
        acq_func: Acquisition function used by the optimiser
        random_state: Seed for the PRNG

    Returns:
        A dict containing the optimisation results, structured as follows:
        'params': {'a': 2.5, 'c': ...} # the optimised params
        'opt_data': {'duration': time taken for opt, 'f': final value of f}
        'opt_params': {'acq_func': ..., } # optimisation parameters

    """
    param_bounds = list(params.values())
    param_names = list(params.keys())

    _f = wrap_function_with_named_args(f, param_names)

    logging.info("Starting optimisation...")
    start = time.time()

    res = gp_minimize(_f,
                      param_bounds,
                      acq_func=acq_func,
                      n_calls=n_calls,
                      n_random_starts=n_random_starts,
                      random_state=random_state)

    end = time.time()
    duration = end - start

    logging.info("Finished optimisation after %.1fs with f=%f." % (duration, res['fun']))

    optimized_parameters = {}
    for i in range(len(param_names)):
        optimized_parameters[param_names[i]] = res['x'][i]

    results = {
        'params': optimized_parameters,
        'opt_data': {'duration': duration, 'f': res['fun']},
        'opt_params': {'acq_func': acq_func,
                       'n_random_starts': n_random_starts,
                       'n_calls': n_calls,
                       'random_state': random_state}
    }

    return results


def wrap_function_with_named_args(f, param_names):

    def _f(vector):
        args = {}
        for i in range(len(param_names)):
            args[param_names[i]] = vector[i]

        return f(**args)

    return _f
