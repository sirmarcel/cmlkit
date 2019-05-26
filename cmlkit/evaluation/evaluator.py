"""The abstract base class Evaluator."""

from cmlkit.engine import Component
from cmlkit import from_config


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
    In this particular case, we the return dict to at least contain
    a key matching the name of a loss function.

    In addition, in *may* contain:
        "lossname_var": The variance for a given loss (when doing CV for instance)
        "refined_config": If the model is refined during evaluation, a configuration
            that will take precedence over the evaluated original config in the search.

    """

    kind = "evaluator"

    def __call__(self, model):
        model = from_config(model, context=self.context)

        return self.evaluate(model)

    def evaluate(self, model):
        raise NotImplementedError
