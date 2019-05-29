"""The abstract base class Evaluator."""

from cmlkit.engine import Component
from cmlkit import from_config
from cmlkit.utility import timed


class Evaluator(Component):
    """Evaluator base class.

    Evaluators exist to evaluate models, usually by training them and
    then computing a loss. The exact details will very depending on
    the use case, so we define only an abstract interface here that
    can be adapated to various uses.

    In general and Evaluator provides a mapping

        config -> dict

    Where dict contains the results of the evaluation, at the moment
    there are no hard expectations on what this dict must hold.

    ***

    One important use for Evaluators is during hyper-parameter tuning.

    In this particular case, there are some expectations on what an
    Evaluator should do. Please check the docstring of `cmlkit.tune`
    for details.

    """

    kind = "evaluator"

    def __call__(self, model):
        model = from_config(model, context=self.context)

        result, duration = timed(self.evaluate)(model)
        result["duration"] = duration

        return result

    def evaluate(self, model):
        raise NotImplementedError
