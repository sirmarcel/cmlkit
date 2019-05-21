# from .mbtr import MBTR1, MBTR2, MBTR3, MBTR4
from .representation import Representation
from .composed import ComposedRepresentation
# from .soap import Soap

classes = {
    # MBTR1.kind: MBTR1,
    # MBTR2.kind: MBTR2,
    # MBTR3.kind: MBTR3,
    # MBTR4.kind: MBTR4,
    ComposedRepresentation.kind: ComposedRepresentation,
    # OnlyCoords.kind: OnlyCoords,
    # OnlyDists.kind: OnlyDists,
    # OnlyDistsHistogram.kind: OnlyDistsHistogram,
    # CoulombMatrixMinimal.kind: CoulombMatrixMinimal,
    # BasicSymmetryFunctions.kind: BasicSymmetryFunctions,
    # EmpiricalSymmetryFunctions.kind: EmpiricalSymmetryFunctions,
}
