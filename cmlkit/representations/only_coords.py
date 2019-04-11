import numpy as np

from ..engine import BaseComponent


class OnlyCoords(BaseComponent):
    """A representation which is just all coordinates

    This will not work for multiple molecules/structures,
    it is purely for conformations.
    """

    kind = 'only_coords'

    def __init__(self, context={}):
        super().__init__(context=context)

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(context=context)

    def _get_config(self):
        return {}

    def compute(self, data):
        return data.r.reshape(data.r.shape[0], -1)
