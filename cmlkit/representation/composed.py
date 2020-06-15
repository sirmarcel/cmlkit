import numpy as np
from cmlkit import classes

from cmlkit import from_config

from .representation import Representation
from .data import GlobalRepresentation, AtomicRepresentation


class Composed(Representation):
    """A representation composed of other representations; results are concatenated"""

    kind = "composed"

    def __init__(self, *reps, context={}):
        super().__init__(context=context)

        self.reps = [from_config(rep, context=self.context) for rep in reps]

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(*config["reps"], context=context)

    def _get_config(self):
        return {"reps": [rep.get_config() for rep in self.reps]}

    def compute(self, data):
        to_concatenate = [rep(data) for rep in self.reps]

        if all(
            [isinstance(rep, GlobalRepresentation) for rep in to_concatenate]
        ):
            computed_representation = np.concatenate(
                [rep.array for rep in to_concatenate], axis=1
            )
            return GlobalRepresentation.from_array(
                self, data, computed_representation
            )
        elif all(
            [isinstance(rep, AtomicRepresentation) for rep in to_concatenate]
        ):
            computed_representation = np.concatenate(
                [rep.linear for rep in to_concatenate], axis=1
            )
            return AtomicRepresentation.from_linear(
                self, data, computed_representation
            )
        else:
            raise ValueError(
                f"Composed representation can either deal with all atomic or all global representations, not mixed cases."
            )
