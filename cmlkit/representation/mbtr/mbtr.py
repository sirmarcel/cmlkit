"""MBTR Base class.

This defines the interface between `cmlkit` and the
MBTR-computing parts of `qmmlpack`.

"""

from cmlkit.representation import Representation
from cmlkit.engine import _from_config, parse_config
from .norm import classes as norms

from cmlkit.utility import import_qmmlpack


class MBTR(Representation):
    """MBTR Base Class.

    The Many-Body Tensor Representation, the MBTR,
    is essentially a broadened histogram of the values of a
    k-body "geometry function", with contributions
    weighted by a "weighting functions".

    For details, please see the `qmmlpack` documentations,
    and the paper https://arxiv.org/abs/1704.06439.

    Please note that this interface has slightly different
    syntax compared to raw `qmmlpack`.

    Parameters:
        start: Value of the first MBTR bin
        stop: Value of last bin
        num: Number of bins
        geomf: String specifying the geometry function
            (for options, see class docstring)
        weightf: String specifying the weighting function (effectively setting the cutoff)
            (for options, see class docstring)
        broadening: Value of exponential broadening.
            (smaller values -> smaller width)
        eindexf: Either "full" or "noreversals", the latter
            prevents redundant counting of elements
        aindexf: Either "full" or "noreversals", the latter
            prevents redundant counting of atoms
        flatten: If True, return flattened MBTR
            (do NOT set to False unless you know what you're doing)
        elems: Elements to include. If None, this will be automatically
            inferred from the dataset when passed. It is recommended to pass
            this explicitly.
        acc: Float specifying when to stop counting contributions.
        norm: A config specifying the normalisation method to be applied.
            (see `mbtr.norm` for options.)

    """

    k = 0

    def __init__(
        self,
        start,
        stop,
        num,
        geomf,
        weightf,
        broadening,
        eindexf="full",
        aindexf="full",
        flatten=True,
        elems=None,
        acc=0.01,
        norm={"none": {}},
        context={},
    ):
        super().__init__(context=context)

        # TODO: insert argument checking here.

        self.mbtr_config = {
            "start": float(start),
            "stop": float(stop),
            "num": int(num),
            "geomf": geomf,
            "weightf": weightf,
            "broadening": float(broadening),
            "eindexf": eindexf,
            "aindexf": aindexf,
            "flatten": flatten,
            "elems": elems,
            "acc": float(acc),
        }

        self.norm = _from_config(norm, classes=norms)

    def compute(self, data):
        mbtr = compute_mbtr_all_k(self.__class__.k, data, self.mbtr_config)

        return self.norm(mbtr)

    def _get_config(self):
        return {**self.mbtr_config, "norm": self.norm.get_config()}


def compute_mbtr_all_k(k, data, mbtr_config):
    """Compute the MBTR."""

    kgwdcea = (
        k,
        mbtr_config["geomf"],
        _to_weightf(mbtr_config["weightf"]),
        _to_distrf(mbtr_config["broadening"]),
        "identity",
        mbtr_config["eindexf"],
        mbtr_config["aindexf"],
    )

    start = mbtr_config["start"]
    stop = mbtr_config["stop"]
    num = mbtr_config["num"]
    d = (start, (stop - start) / num, num)

    # todo: insert dispatch to wrapped call
    # in subprocess here.
    qmmlpack = import_qmmlpack("compute MBTR (not using dscribe)")
    mbtr = qmmlpack.many_body_tensor(
        data.z,
        data.r,
        d,
        kgwdcea,
        basis=data.b,
        elems=mbtr_config["elems"],
        acc=mbtr_config["acc"],
        flatten=mbtr_config["flatten"],
    )

    return mbtr


def _to_weightf(config):
    """Translate to qmmlpack weightf format."""

    if isinstance(config, str):
        return config
    else:
        kind, inner = parse_config(config)

        return (kind, (inner["ls"],))


def _to_distrf(broadening):
    """ Translate to qmmlpack distrf format."""
    return ("normal", (broadening,))
