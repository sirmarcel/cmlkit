from .representation import Representation

from cmlkit.utility import import_qmmlpack


def compute_cm(data, config):

    assert data.b is None, "Coulomb matrix cannot handle periodic systems!"

    qmmlpack = import_qmmlpack("compute coulomb matrix")
    return qmmlpack.coulomb_matrix(
        data.z,
        data.r,
        unit=config["unit"],
        padding=config["padding"],
        flatten=config["flatten"],
        sort=config["sort"],
    )


class CoulombMatrix(Representation):
    """Coulomb Matrix Representation.

    Introduced in
        Rupp, Tkatchenko, Müller, von Lilienfeld, PRL 108, 058301 (2012).

    The coulomb matrix is essentially a matrix constructed from charges and
    inverse distances. To make it work across structures, it needs to be padded,
    and sorted.

    Parameters:
        unit: default is "BohrRadius", alternatively can be "Picometers" or "Angstrom".
            (I don't really know why there are units here -- BohrRadius does not convert the
            input, which seems to be the correct approach to me.)
        padding: if not False, int specifying the size of the matrix (must be the maximum
            number of atoms per structure in the dataset to make sense)
        sorting: if True, sort the CM by norm
        flatten: if True, return vector version (must be True in most cases, `cmlkit` is
            not explicitly built to handle non-flat representations)

    """

    kind = "cm"

    def __init__(
        self, unit="BohrRadius", padding=False, flatten=True, sort=True, context={}
    ):
        super().__init__(context=context)

        self.cm_config = {
            "unit": unit,
            "padding": padding,
            "flatten": flatten,
            "sort": sort,
        }

    def _get_config(self):
        return self.cm_config

    def compute(self, data):
        return compute_cm(data, self.cm_config)
