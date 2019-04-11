import numpy as np
from cmlkit import from_config
from ..engine import BaseComponent


class ComposedRepresentation(BaseComponent):
    """A representation composed of other representations; results are concatenated"""

    kind = 'composed'

    def __init__(self, *reps, context={}):
        super().__init__(context=context)
        self.reps = [cml.from_config(rep, context=self.context) for rep in reps]

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(*config['reps'], context=context)

    def _get_config(self):
        return {'reps': [rep.get_config() for rep in self.reps]}

    def compute(self, data):
        to_concatenate = [rep.compute(data) for rep in self.reps]

        return np.concatenate(to_concatenate, axis=1)
