"""Regression Data sub-classes."""

from cmlkit.engine import Data


class KernelMatrix(Data):

    kind = "data_kernel_matrix"

    @classmethod
    def from_array(cls, kernel, representation, array):
        data = {"array": array}

        return cls.result(data=data, inputs=representation, component=kernel)

    @property
    def array(self):
        return self.data["array"]

    @classmethod
    def mock(cls, array):
        return cls.create(data={"array": array})
