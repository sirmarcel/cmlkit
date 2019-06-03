"""Search algorithms.

In `cmlkit.tune`, a search algorithm is expected to emit
trial configs and receive losses in return, based on
which it can then suggest further trials. It should be
okay with receiving these trials back in random order,
and must be deterministic if suggestion and submission
of results happen in precisely the same sequence again.

Searches must therefore implement a suggest and submit method.

Since we currently only have hyperopt, there is no abstract base class.

Nomenclature:
    trial config: suggestion emitted by search
    tid: id of that trial, how the result will be tagged
        when returned to the search


"""

from .hyperopt import Hyperopt
