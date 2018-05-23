import numpy as np
import os
from unittest import TestCase
from unittest.mock import MagicMock
import qmmltools.inout as qmtio
from qmmltools.mbtr.mbtr import MBTR


dirname = os.path.dirname(os.path.abspath(__file__))


class TestMBTR(TestCase):
    def setUp(self):
        # load test data
        self.data = qmtio.read(dirname + '/test_model_mini.mbtr')

    def test_from_file(self):
        mbtr = MBTR.from_file(dirname + '/test_model_mini.mbtr.npy')

        self.assertEqual(self.data['name'], mbtr.name)
        np.testing.assert_array_equal(self.data['mbtr'], mbtr.mbtr)
        self.assertEqual(self.data['spec'], mbtr.spec)
        self.assertEqual(self.data['spec_name'], mbtr.spec_name)
        self.assertEqual(self.data['dataset_id'], mbtr.dataset_id)
