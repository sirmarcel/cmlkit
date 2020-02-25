import numpy as np
from cmlkit import classes

from .representation import Representation
from cmlkit import from_config


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

        # hack alert! there is no well-defined way to recognise
        # which rep is local/global
        is_local = to_concatenate[0].dtype == object

        if not is_local:
            return np.concatenate(to_concatenate, axis=1)
        else:
            # local reps are object arrays of a single list, so they don't
            # have a first axis -- this is just an ugly workaround

            return np.array(
                [
                    np.concatenate([rep[i] for rep in to_concatenate], axis=1)
                    for i in range(data.n)
                ],
                dtype=object,
            )
