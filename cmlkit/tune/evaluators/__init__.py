"""Collection of evaluators specialised for hyperparameter tuning.

This is somewhat awkward, but it makes some sense: Tune-focused
evaluators should only ever compute a single loss, not multiple
losses, and they need to be optimised for speed above all else.

"""

from .evaluator_holdout import TuneEvaluatorHoldout
