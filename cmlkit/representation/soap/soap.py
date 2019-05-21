"""SOAP Representation."""


from cmlkit.representation import Representation
from .quippy_interface import compute_soap


class SOAP(Representation):
    """SOAP Representation.

    The SOAP representation was originally introduced in

        Bartok, Kondor, Csanyi, PRB 87, 184115 (2013).

    (But grew out of previous work by the authors on
    the bi-spectrum, etc.)

    It is an atomic representation that:
    - Broadens the positions of neighbour atoms (within a cutoff)
      with Gaussians.
    - Expands this density in spherical harmonics and a radial basis.
    - Introduces rotational symmetry by combining the resulting expansion
      coefficients in a specific way, into a power-/bi-spectrum.

    (An alternative perspective is that of integrating out rotations.)

    What we call the SOAP representation here is the power spectrum
    from the paper cited above. For more than one element, we have a
    series of power spectra organised by elements.

    In more pragmatic terms, the SOAP representation is whatever quippy produces when
    you ask it to compute the soap descriptor.

    Parameters:
        elems: Elements for which we compute SOAP
        cutoff: Cutoff radius
        sigma: Broadening
        n_max: Number of radial basis functions
        l_max: Number of angular basis functions

    """

    kind = "soap"
    default_context = {"cleanup": True, "timeout": None}

    def __init__(self, elems, cutoff, sigma, n_max, l_max, context={}):
        super().__init__(context=context)

        self.quippy_config = {
            "elems": elems,
            "cutoff": cutoff,
            "sigma": sigma,
            "n_max": n_max,
            "l_max": l_max,
        }

    def compute(self, data):
        return compute_soap(
            data,
            self.quippy_config,
            cleanup=self.context["cleanup"],
            timeout=self.context["timeout"],
        )

    def _get_config(self):
        return self.quippy_config
