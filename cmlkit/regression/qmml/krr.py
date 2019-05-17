"""Regressor implementing KRR."""

import qmmlpack

from cmlkit.engine import Component
from cmlkit import from_config


class KRR(Component):
    """Kernel Ridge Regression with qmmlpack backend."""

    kind = "krr"
    default_context = {"print_timings": False}

    def __init__(self, kernel, nl, centering=False, context={}):
        super().__init__(context=context)

        self.kernel = from_config(kernel, context=self.context)
        self.nl = nl
        self.centering = centering

        self.trained = False

    def _get_config(self):
        return {
            "nl": self.nl,
            "kernel": self.kernel.get_config(),
            "centering": self.centering,
        }

    def train(self, x, y):
        """Train KRR model.

        Args:
            x: Either global or atomic representations.
            y: Array with labels.

        """
        self.x_train = x

        kernel = self.kernel(self.x_train)

        self.krr = qmmlpack.KernelRidgeRegression(
            kernel, y, theta=(self.nl,), centering=self.centering
        )

        self.trained = True
        return self  # return the trained regressor!

    def predict(self, z):
        """Predict with KRR model.

        Args:
            z: Either global or atomic representations.
        """

        kernel = self.kernel(x=self.x_train, z=z)

        prediction = self.krr(kernel)

        return prediction
