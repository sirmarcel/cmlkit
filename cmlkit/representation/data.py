"""Data classes for computed representations."""

import numpy as np
from cmlkit.engine import Data


class GlobalRepresentation(Data):
    kind = "data_global_representation"

    @classmethod
    def from_array(cls, representation, dataset, array):
        data = {"array": array}

        return cls.result(
            data=data, input_data=dataset, component=representation
        )

    @classmethod
    def mock(cls, array):
        return cls.create(data={"array": array})

    @property
    def array(self):
        return self.data["array"]


class AtomicRepresentation(Data):
    kind = "data_atomic_representation"

    @classmethod
    def from_linear(cls, representation, dataset, linear):
        data = atomic_data_dict(
            dataset.info["atoms_by_system"], linear
        )

        return cls.result(
            data=data, input_data=dataset, component=representation
        )

    @classmethod
    def from_ragged(cls, representation, dataset, ragged):
        linear = np.concatenate(ragged, axis=0)
        return cls.from_linear(representation, dataset, linear)

    @classmethod
    def mock(cls, counts, linear):
        data = atomic_data_dict(counts, linear)
        return cls.create(data=data)

    @property
    def n(self):
        return len(self.offsets) - 1

    @property
    def linear(self):
        return self.data["linear"]

    @property
    def offsets(self):
        return self.data["offsets"]

    @property
    def counts(self):
        return self.data["counts"]

    @property
    def ragged(self):
        return np.array(
            [
                self.linear[self.offsets[i] : self.offsets[i + 1]]
                for i in range(self.n)
            ],
            dtype=object,
        )


def atomic_data_dict(counts, linear):
    offsets = np.zeros(len(counts) + 1, dtype=int)
    offsets[1::] = np.cumsum(counts)

    data = {
        "linear": linear,
        "offsets": offsets,
        "counts": counts,
    }

    return data
