"""Exceptions that can be automatically caught by tune.

As far as I know, there is no straightforward way to properly serialise exceptions.
So we're doing it my hand for now -- open to suggestions on how to do it better.
"""

from qmmlpack import QMMLException
from concurrent.futures import TimeoutError

exceptions = {
    "QMMLException": QMMLException,
    "TimeoutError": TimeoutError,
    "ValueError": ValueError,
    "AssertionError": AssertionError,
    "RuntimeError": RuntimeError,
    "Exception": Exception,  # this will catch basically everything
}


def get_exceptions(args):
    result = []
    for a in args:
        if a in exceptions:
            result.append(exceptions[a])
        else:
            raise ValueError(
                f"cmlkit.tune can currently not catch {a}. Please file an issue."
            )

    return tuple(result)  # except expects a tuple, not a list


def get_exceptions_spec(args):
    return [a.__name__ for a in args]
