"""Symmetry functions."""

from cmlkit.representation import Representation
from .config import prepare_config
from .runner import compute_symmfs
from .runner_input import make_infile


class SymmetryFunctions(Representation):
    """Atom-Centered Symmetry Functions.

    Symmetry functions are an atomic representation,
    consisting of an array of function values computed
    with different parametrisations.

    They were introduced by Jörg Behler in
    Behler, JCP 134, 074106 (2011).

    At the moment, we support only the "standard"
    angular and radial symmetry functions, which are:

    "rad": G_i^2 (eq. 6) in the paper.
    Parameters:
        cutoff: Maximum distance to central atom.
        eta: Width of the Gaussian.
        mu: Shift of center. (R_S in equation.)

    "ang": G_i^4 (eq. 8) in the paper.
    Parameters:
        cutoff: Maximum distance to central atom.
        eta: Width of Gaussian.
        zeta: Roughly, the width of the angular distribution.
        lambd: "Direction" of angular distribution. (Only +-1.)
            (not that `lambda` is a keyword, so it's lambd)

    For each of these, there is a "universal" version, which is applied
    to all element combinations with the same parameters, and a more
    specific "elemental" version, where the elements can be specified
    by hand.

    This implementation allows you to specify both, individual
    symmetry functions (i.e. with all parameters set already),
    or to just give a parametrisation scheme that sets (some) of the
    parameters automatically. (See `parametrization.py`.)

    Parameters:
        elems: List of elements for which SFs are supposed to be computed
        elemental: List of configs of elemental SFs
        universal: List of configs of universal SFs or configs of parametrization
            schemes (for options, see `parametriziation.py`.
        dim: Dimensionality of resulting descriptor (will be inferred if None given)

    """

    kind = "sf"
    default_context = {"cleanup": True, "timeout": None}

    def __init__(self, elems, elemental=[], universal=[], dim=None, context={}):
        super().__init__(context=context)
        self.runner_config = prepare_config(elems, elemental, universal, dim)
        self.config = {
            "elems": elems,
            "elemental": elemental,
            "universal": universal,
            "dim": dim,
        }

    def compute(self, data):
        return compute_symmfs(
            data,
            self.runner_config,
            cleanup=self.context["cleanup"],
            timeout=self.context["timeout"],
        )

    def _get_config(self):
        return self.config

    def get_infile(self):
        return make_infile(self.runner_config)
