"""Evaluate model on a holdout dataset."""

from cmlkit import load_dataset

from cmlkit.evaluation.evaluator import Evaluator
from cmlkit.evaluation.loss import get_lossf


class TuneEvaluatorHoldout(Evaluator):
    """Evaluate model on holdout set, specialised for parameter tuning.

    Trains the model on a training set, then computes the
    loss for an unseen test set.

    Parameters:
        train: training dataset
        test: test dataset
        target: name of target quantity (must be present in train and test)
        per: unit of quantity (per atom? per molecule?)
        lossf: name of a loss function

    """

    kind = "tune_eval_holdout"

    def __init__(self, train, test, target, per=None, lossf="rmse", context={}):
        super().__init__(context=context)

        self.train = load_dataset(train)
        self.test = load_dataset(test)
        self.lossf = get_lossf(lossf)
        self.target = target
        self.per = per

    def _get_config(self):
        return {
            "train": self.train.name,
            "test": self.test.name,
            "lossf": self.lossf.__name__,
            "per": self.per,
            "target": self.target,
        }

    def evaluate(self, model):
        model.train(self.train, target=self.target)
        pred = model.predict(self.test, per=self.per)
        return {"loss": self.lossf(self.test.pp(self.target, per=self.per), pred)}
