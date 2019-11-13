"""MBTR normalisation schemes."""

import numpy as np

from cmlkit.engine import Component


class SimpleNorm(Component):

    kind = "simple"

    def __init__(self, scale=1.0, context={}):
        super().__init__(context=context)

        self.scale = scale

    def __call__(self, mbtr, data=None):
        # avoid spamming the log with divide by zero warnings
        with np.errstate(divide="ignore"):
            mbtr /= np.sum(
                mbtr, axis=1, keepdims=True
            )  # divide by overall sum of MBTR bins

        scaled = mbtr * self.scale
        scaled = np.nan_to_num(scaled, copy=False)

        return scaled

    def _get_config(self):
        return {"scale": self.scale}


class L2Norm(Component):
    """Normalise by l2 norm."""

    kind = "l2"

    def __init__(self, scale=1.0, context={}):
        super().__init__(context=context)

        self.scale = scale

    def __call__(self, mbtr, data=None):
        with np.errstate(divide="ignore"):
            mbtr /= np.linalg.norm(mbtr, axis=1, keepdims=True, ord=2)
            mbtr *= self.scale

        return mbtr

    def _get_config(self):
        return {"scale": self.scale}


class NoneNorm(Component):

    kind = "none"

    def __call__(self, mbtr, data=None):
        return mbtr

    def _get_config(self):
        return {}


classes = {SimpleNorm.kind: SimpleNorm, NoneNorm.kind: NoneNorm, L2Norm.kind: L2Norm}
