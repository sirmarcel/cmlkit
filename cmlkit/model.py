"""Model class.

A Model is a combination of representation and regression method,
and can be regarded as a rudimentary pipeline.

It essentially wraps these two components, and takes care
of passing the computed representation around to the regressor.

The only additional task a Model has is to make sure the property
to be predicted is converted appropriately for the regression method.

For instance, it is occasionally better to predict a target quantity
normalised by the number of atoms in a structure, or the convention
in a community demands per-atom predictions, but the model is better
suited to predict quantitites for the entire system.

The `cmlkit` convention is that all properties are stored scaled
with the number of atoms in the system. This is arbitrary, but it
makes the conversion a bit more easy.

An alternative approach is to always use `None` as `per`, in which
case no conversion is ever done!

"""

from cmlkit.engine import Component
from cmlkit import from_config
from cmlkit.representation import Composed
from .utility import convert, unconvert


class Model(Component):
    """Model class.

    When training, automatically computes a representation,
    and then trains a regression method. When predicting,
    automatically computes the representation for that,
    and then predicts using the regression method.

    Attributes:
        representation: Representation instance.
        regression: Regression method.
        per: Preferred units/scaling for target property.
            (Popular choices: "atom", "cell", "mol")
            (see `conversion.py` for more info.)

    """

    kind = "model"

    def __init__(self, representation, regression, per=None, context={}):
        """Create model.

        Args:
            representation: Representation instance, or config for one, or
                a list with any of the above, in which case a Composed representation
                is automatically generated.
            regression: Regression method or config of one.
            per: Optional, String (or None) specifying per what the regression should
                internally predict. Default is to not convert.

        """
        super().__init__(context=context)

        # Allowing myself ONE piece of "magic"!
        if isinstance(representation, (list, tuple)):
            self.representation = Composed(*representation, context=self.context)
        else:
            self.representation = from_config(representation, context=self.context)

        self.regression = from_config(regression, context=self.context)

        self.per = per

    def _get_config(self):
        return {
            "representation": self.representation.get_config(),
            "regression": self.regression.get_config(),
            "per": self.per,
        }

    def train(self, data, target):
        """Train model.

        Args:
            data: Dataset instance
            target: Name of target property,
                must be present in data.
        """
        x = self.representation(data)
        y = data.pp(target, self.per)

        self.regression.train(x=x, y=y)

        return self  # return trained Model

    def predict(self, data, per=None):
        """Predict with model.

        Args:
            data: Dataset instance
            per: Optional, String specifying in which units
                the prediciton should be made.

        Returns:
            ndarray with predictions.

        """
        z = self.representation(data)
        pred = self.regression.predict(z)
        pred = unconvert(data, pred, from_per=self.per)
        return convert(data, pred, per)
