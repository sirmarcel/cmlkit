"""Exceptions that can be automatically caught by tune.

As far as I know, there is no straightforward way to properly serialise exceptions.
So we're doing it by hand for now -- open to suggestions on how to do it better.
"""

from concurrent.futures import TimeoutError

exceptions = {
    "TimeoutError": TimeoutError,
    "ValueError": ValueError,
    "AssertionError": AssertionError,
    "RuntimeError": RuntimeError,
    "Exception": Exception,  # this will catch basically everything
}

try:
    from qmmlpack import QMMLException
    exceptions["QMMLException"] = QMMLException
except ImportError:
    # Rare instance where it's ok to do this!
    pass


def get_exceptions(args):
    result = []
    for a in args:
        if a in exceptions:
            result.append(exceptions[a])
        else:
            raise ValueError(
                f"cmlkit.tune can currently not catch {a}. Please file an issue. (QMMLException can be caught if qmmlpack is installed.)∫"
            )

    return tuple(result)  # except expects a tuple, not a list


def get_exceptions_spec(args):
    return [a.__name__ for a in args]
