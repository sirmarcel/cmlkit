"""Evaluate model on a holdout dataset."""

from cmlkit import load_dataset

from .evaluator import Evaluator
from .loss import get_loss


class EvaluatorHoldout(Evaluator):
    """Evaluate model on holdout set.

    Trains the model on a training set, then computes the
    loss for an unseen test set.

    Parameters:
        train: training dataset
        test: test dataset
        target: name of target quantity (must be present in train and test)
        per: unit of quantity (per atom? per molecule?)
        loss: loss specification (see `loss` module for details.)

    """

    kind = "eval_holdout"

    def __init__(self, train, test, target, per=None, loss="rmse", context={}):
        super().__init__(context=context)

        self.train = load_dataset(train)
        self.test = load_dataset(test)
        self.loss = get_loss(loss)
        self.target = target
        self.per = per

    def _get_config(self):
        return {
            "train": self.train.name,
            "test": self.test.name,
            "loss": self.loss.spec,
            "per": self.per,
            "target": self.target,
        }

    def evaluate(self, model):
        model.train(self.train, target=self.target)
        pred = model.predict(self.test, per=self.per)

        return self.loss(self.test.pp(self.target, per=self.per), pred)
