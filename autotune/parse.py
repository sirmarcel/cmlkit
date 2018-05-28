import copy
from hyperopt import hp
import qmmltools.inout as qmtio
import qmmltools.helpers as qmth
import qmmltools.stats as qmts


def parse(d):
    """Convert a dict conveniently writeable settings into the 'real' ones.

    Syntax:
        'loss': 'str' -> 'loss': qmts.str
        ('grid_log2', min, max, n) -> np.logspace(...)
        ('hp_func', 'id', arg) -> hp.func('id', arg)

    In particular, the following operations are performed:
        - Convert losses into functions (from strings)
        - Generate grids in-place
        - Find hyperopt functions and create them

    """

    qmth.find_key_apply_f(d, 'loss', string_to_loss)
    qmth.find_pattern_apply_f(d, is_hyperopt, to_hyperopt)


def string_to_loss(s):

    try:
        f = getattr(qmts, s)
    except AttributeError:
        raise NotImplementedError("Loss named {} is not implemented.".format(s))

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
