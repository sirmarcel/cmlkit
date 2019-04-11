"""This module implements various classes that evaluate Models"""

from .eval_base import EvaluatorBase
from .eval_holdout import EvaluatorHoldout
from .eval_lgs import EvaluatorLGS
from .eval_cv import EvaluatorCV

classes = {
    EvaluatorBase.kind: EvaluatorBase,
    EvaluatorHoldout.kind: EvaluatorHoldout,
    EvaluatorLGS.kind: EvaluatorLGS,
    EvaluatorCV.kind: EvaluatorCV,
}
