"""Infrastructure for the evaluation of models."""

from .loss import get_loss, get_lossf
from .evaluator_holdout import EvaluatorHoldout

components = [EvaluatorHoldout]
