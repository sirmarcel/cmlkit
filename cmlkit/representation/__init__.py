from .representation import Representation
from .composed import Composed
from .soap import SOAP
from .mbtr import MBTR1, MBTR2, MBTR3, MBTR4
from .sf import SymmetryFunctions
from .coulomb_matrix import CoulombMatrix

from .soap import classes as classes_soap
from .mbtr import classes as classes_mbtr
from .sf import classes as classes_sf

classes = {
    **classes_soap,
    **classes_mbtr,
    **classes_sf,
    CoulombMatrix.kind: CoulombMatrix,
}
