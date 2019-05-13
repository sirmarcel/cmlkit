import time
from qmmlpack import local_grid_search

from cmlkit import logger
import cmlkit.helpers as cmlh
from cmlkit.engine import Component


class LocalGridSearch(Component):
    """Wrapper for qmmlpack global grid search"""
    kind = 'lgs'

    def __init__(self, seed=None, resolution=0.0001, maxevals=50, ignore=[], silent=True, context={}):
        super().__init__(context=context)
        self.seed = seed
        self.resolution = resolution
        self.maxevals = maxevals
        self.ignore = ignore
        self.silent = silent

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {'seed': self.seed,
                'resolution': self.resolution,
                'maxevals': self.maxevals,
                'ignore': self.ignore,
                'silent': self.silent}

    def needs_lgs(self, template):
        """Can LGS be applied to this?"""
        locations = cmlh.find_pattern(template, lgs_pattern, ignore=self.ignore)
        return len(locations) > 0

    def optimize(self, template, objective):
        return run_lgs(template, objective,
                       seed=self.seed,
                       resolution=self.resolution,
                       maxevals=self.maxevals,
                       ignore=self.ignore,
                       silent=self.silent)


def run_lgs(template, objective, seed=None, resolution=0.0001, maxevals=50, ignore=[], silent=False):
    """Run a logspace local grid search.

    This optimises a function that expects a dict/list/tuple nested data
    structure as input. Parameters to optimise are indicated by a list (or
    tuple) of the form

        ['lgs', (arguments for local_grid_search)]

    The second item must follow the conventions laid out in the docs for
    qmmlpack.local_grid_search. The optimised parameters are substituted
    in place of the tokens, so note that template is changed!

    Information during optimisation is written to the logger.

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
    start = time.time()

    cmlh.tuples_to_lists(template)

    wrapped_objective, variables, locations = prepare_lgs(template, objective, ignore=ignore)

    logger.debug("Running grid search for the following variables: {}".format(locations))
    if not silent:
        logger.info("Starting local grid search with {} variables.".format(len(locations)))

    result = local_grid_search(f=wrapped_objective,
                               variables=variables,
                               evalmonitor=log_during_eval,
                               resolution=resolution,
                               maxevals=maxevals,
                               seed=seed,
                               )

    best_values = result['best_valpow']
    best_f = result['best_f']

    if not silent:
        logger.info("Ended local grid search with loss {}.".format(best_f))

    for i, arg in enumerate(best_values):
        loc = locations[i]
        cmlh.set_with_path(template, loc, arg)

    end = time.time()
    duration = end - start
    results = {'best_values': best_values,
               'locations': locations,
               'duration': duration,
               'num_evals': result['num_evals'],
               }

    return results


def log_during_eval(trialv, trialf, bestv, bestf, state):
    logger.debug("Step {}/{}, f={} ({})".format(state['num_evals'], state['max_evals'], trialf, bestf))


def lgs_pattern(x):
    if isinstance(x, (tuple, list)):
        if len(x) > 0:  # this guards against empty arrays
            return x[0] == 'lgs'
        else:
            return False

    return False


def prepare_lgs(template, objective, ignore=[]):
    locations = cmlh.find_pattern(template, lgs_pattern, ignore=ignore)
    variables = [cmlh.get_with_path(template, loc)[1] for loc in locations]

    def wrapped_objective(*args):
        for i, arg in enumerate(args):
            loc = locations[i]
            cmlh.set_with_path(template, loc, arg)

        return objective(template)

    return wrapped_objective, variables, locations
