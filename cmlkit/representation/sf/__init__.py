"""Interface with RuNNer to compute Behler symmetry functions.

This submodule is organised as follows:

`runner_*.py` takes care of executing runner, taking in specifications
of the symmetry functions to be computed in an intermediate config dict.

This dict is generated in `config.py`, potentially using parametrisation
schemes implemented in `parametrization.py`.

The Representation itself is implemented in `sf.py`.

For information on the syntax, see the class definition in `sf.py`,
or `config.py`. Thanks!

Note that computing symmetry functions requires a functioning binary of
RuNNer to be referenced in the `$CML_RUNNER_PATH` environment variable.

"""

from .sf import SymmetryFunctions
from .runner import compute_symmfs
from .config import prepare_config

components = [SymmetryFunctions]
