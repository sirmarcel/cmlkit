from unittest import TestCase
from unittest.mock import MagicMock, PropertyMock
import numpy as np

from cmlkit.representation.data import (
    GlobalRepresentation,
    AtomicRepresentation,
)


class TestAtomicRepresentation(TestCase):
    def setUp(self):
        self.counts = [3, 2, 1]
        self.linear = np.random.random((6, 5))
        self.ragged = np.array(
            [self.linear[0:3, :], self.linear[3:5, :], self.linear[[5], :]]
        )

        self.component = MagicMock()
        self.component.get_hid = MagicMock(return_value="anything")

        self.dataset = MagicMock()
        history = PropertyMock(return_value=["dataset"])
        info = PropertyMock(return_value={"atoms_by_system": self.counts})
        type(self.dataset).info = info
        type(self.dataset).history = history

    def test_history(self):
        rep = AtomicRepresentation.from_linear(
            self.component, self.dataset, self.linear
        )

        self.assertEqual(len(rep.history), 2)

    def test_from_linear(self):
        rep = AtomicRepresentation.from_linear(
            self.component, self.dataset, self.linear
        )

        for i in range(3):
            np.testing.assert_array_equal(rep.ragged[i], self.ragged[i])

        np.testing.assert_array_equal(rep.linear, self.linear)

    def test_from_ragged(self):
        rep = AtomicRepresentation.from_ragged(
            self.component, self.dataset, self.ragged
        )

        np.testing.assert_array_equal(rep.linear, self.linear)

        for i in range(3):
            np.testing.assert_array_equal(rep.ragged[i], self.ragged[i])

    def test_range(self):
        rep = AtomicRepresentation.mock(self.counts, self.linear)
        np.testing.assert_array_equal(rep.range((1, 3)).ragged[0], rep.ragged[1])
        np.testing.assert_array_equal(rep.range((1, 3)).ragged[-1], rep.ragged[2])
        np.testing.assert_array_equal(rep.range((1, 3)).ragged[1], rep.ragged[2])


class TestGlobalRepresentation(TestCase):
    def setUp(self):

        self.component = MagicMock()
        self.component.get_hid = MagicMock(return_value="anything")

        self.dataset = MagicMock()
        history = PropertyMock(return_value=["dataset"])
        type(self.dataset).history = history

    def test_works(self):
        data = np.random.random((3, 2))

        rep = GlobalRepresentation.from_array(
            self.component, self.dataset, data
        )

        np.testing.assert_array_equal(rep.array, data)
        self.assertEqual(len(rep.history), 2)
