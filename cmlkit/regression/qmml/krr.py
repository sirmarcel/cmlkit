"""Regressor implementing KRR."""


from cmlkit.engine import Component
from cmlkit import from_config

from cmlkit.utility import import_qmmlpack


class KRR(Component):
    """Kernel Ridge Regression with qmmlpack backend.

    Parameters:
        kernel: Component (or config) with signature f(x, z=None) -> ndarray that
            computes kernel matrices between local and global reps
            (either square for x (making use of symmetry), or between x and z)
        nl: Regularisation strength, equivalent to `sigma**2` in Rasmussen & Williams
            (this is the factor that is added to the diagonal elements of the kernel matrix)
        centering: Optional, if True, labels and kernel matrices are centered to mean=0

    """

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

        kernel = self.kernel(self.x_train).array

        qmmlpack = import_qmmlpack("use cmlkit.regression.qmml")
        self.krr = qmmlpack.KernelRidgeRegression(
            kernel, y, theta=(self.nl,), centering=self.centering
        )

        self.trained = True
        return self  # return the trained regressor!

    def predict(self, z):
        """Predict with KRR model.

        Args:
            z: Either global or atomic representation.
        """

        kernel = self.kernel(x=self.x_train, z=z).array

        prediction = self.krr(kernel)

        return prediction
