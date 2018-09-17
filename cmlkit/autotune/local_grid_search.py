"""Interface code for qmmlpack.local_grid_search"""

import numpy as np
import qmmlpack as qmml
import cmlkit
import cmlkit.helpers as cmlh


def run_lgs(template, objective, resolution=None, maxevals=100, ignore=[]):
    """Run a logspace local grid search.

    This optimises a function that expects a dict/list/tuple nested data
    structure as input. Parameters to optimise are indicated by a list (or
    tuple) of the form

        ['lgs', (arguments for local_grid_search)]

    The second item must follow the conventions laid out in the docs for
    qmmlpack.local_grid_search. The optimised parameters are substituted
    in place of the tokens, so note that template is changed!

    Information during optimisation is written to the cmlkit logger.

    Args:
        template: dict/list/tuple nested data structure
        objective: function to optimise
        resolution: optional, if specified as a number a,
                    objective will be rounded to multiples of a
        maxevals: optional, maximum evaluations before terminating
        ignore: list of paths to ignore in template

    Returns:
        A dict with diagnostic info.

    """

    wrapped_objective, variables, locations = prepare_lgs(template, objective, ignore=ignore)
    cmlkit.logger.debug("Running grid search for the following variables: {}".format(locations))

    cmlkit.logger.info("Starting local grid search with {} variables.".format(len(locations)))
    best_v, best_f = qmml.local_grid_search(wrapped_objective, variables, evalmonitor=log_during_eval,
                                            resolution=resolution, maxevals=maxevals)
    cmlkit.logger.info("Ended local grid search with loss {}.".format(best_f))
    # TODO: Once qmmlpack is updated to return actually optimised values remove this, it is very stupid.
    best_values = np.power(2.0, best_v)

    for i, arg in enumerate(best_values):
        loc = locations[i]
        cmlh.set_with_path(template, loc, arg)

    results = {'best_values': best_values, 'locations': locations}

    return results


def log_during_eval(trialv, trialf, bestv, bestf, state):
    cmlkit.logger.debug("Step {}/{}, f={} ({})".format(state['num_evals'], state['max_evals'], trialf, bestf))


def lgs_pattern(x):
    if isinstance(x, (tuple, list)):
        return x[0] == 'lgs'

    return False


def prepare_lgs(template, objective, ignore=[]):
    locations = cmlh.find_pattern(template, lgs_pattern, ignore=ignore)
    variables = (cmlh.get_with_path(template, loc)[1] for loc in locations)

    def wrapped_objective(*args):
        for i, arg in enumerate(args):
            loc = locations[i]
            cmlh.set_with_path(template, loc, arg)

        return objective(template)

    return wrapped_objective, variables, locations
