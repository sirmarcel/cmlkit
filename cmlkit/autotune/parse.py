import copy
import logging
from hyperopt import hp
import cmlkit.autotune.grid as gr
import cmlkit.inout as cmlio
import cmlkit.helpers as cmlh
import cmlkit.losses as cmls


def preprocess(d):
    og = copy.deepcopy(d)

    # Setup the spec template for later
    d['spec']['name'] = d['name']
    d['spec']['desc'] = 'Model during autotune run; ' + d['desc']

    # Defaults
    defaults_config = {'parallel': False, 'loss': 'rmse', 'n_cands': 2, 'loglevel': 'INFO'}
    d['config'] = {**defaults_config, **d['config']}

    parse(d)

    d['internal'] = {}
    d['internal']['original_task'] = og




def parse(d):
    """Convert a dict conveniently writeable settings into the 'real' ones.

    Syntax:
        'loss': 'str' -> 'loss': cmls.str
        ('gr_log2', min, max, n) -> np.logspace(...)
        ('hp_func', 'id', arg) -> hp.func('id', arg)

    In particular, the following operations are performed:
        - Convert losses into functions (from strings)
        - Generate grids in-place
        - Find hyperopt functions and create them

    """

    cmlh.find_key_apply_f(d, 'loss', string_to_loss)
    cmlh.find_key_apply_f(d, 'loglevel', string_to_loglevel)
    cmlh.find_pattern_apply_f(d, is_grid, to_grid)
    cmlh.find_pattern_apply_f(d, is_hyperopt, to_hyperopt)


def string_to_loss(s):

    try:
        f = getattr(cmls, s)
    except AttributeError:
        raise NotImplementedError("Loss named {} is not implemented.".format(s))

    return f


def string_to_loglevel(s):

    try:
        f = getattr(logging, s)
    except AttributeError:
        raise NotImplementedError("Loglevel named {} cannot be found implemented (should be DEBUG, INFO, ERROR or WARNING).".format(s))

    return f


def is_hyperopt(x):
    """Check whether a given object is a hyperopt argument

    The format expected is ('hp_NAME_OF_FUNCTION', 'name for hyperopt', remaining, arguments)

    """

    if isinstance(x, (tuple, list)):
        if isinstance(x[0], str):
            s = x[0].split('_', 1)
            if s[0] == 'hp':
                return True

    return False


def to_hyperopt(x):
    """Convert a sequence to a hyperopt function

    Example: ('hp_choice', 'mbtr_1', [1, 2, 3])
             -> hp.choice('mbtr_1', [1, 2, 3])

    """

    s = x[0].split('_', 1)
    try:
        f = getattr(hp, s[1])
    except AttributeError:
        raise NotImplementedError("Hyperopt can't find function named {}!".format(s[1]))

    f = f(*x[1:])
    return f


def is_grid(x):
    """Check whether a given object is a grid argument

    The format expected is ('gr_NAME_OF_GRID', remaining, arguments)

    """

    if isinstance(x, (tuple, list)):
        if isinstance(x[0], str):
            s = x[0].split('_', 1)
            if s[0] == 'gr':
                return True

    return False


def to_grid(x):
    """Convert a sequence to a grid

    Supported functions:
        'log2': base 2 grid
        'lin': linear grid

    Example: ('grid_log2', -20, 20, 11)
             -> np.logspace(-20, 20, base=2, num=11)

    """

    s = x[0].split('_', 1)
    try:
        f = getattr(gr, s[1])
    except AttributeError:
        raise NotImplementedError("Grid named {} is not (yet) implemented!".format(s[1]))

    return f(*x[1:])
