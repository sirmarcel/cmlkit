from unittest import TestCase
from unittest.mock import MagicMock
import qmmltools.inout as qmtio
import os
from qmmltools.model import ModelSpec, mbtr_defaults

dirname = os.path.dirname(os.path.abspath(__file__))


class TestModel(TestCase):
    def setUp(self):
        # load test data
        self.data = qmtio.read_yaml(dirname + '/model_1.yaml')


    def test_init_from_dict(self):
        modelSpec = ModelSpec.from_dict(self.data)

        self.assertEqual(self.data['name'], modelSpec.name)
        self.assertEqual(self.data['desc'], modelSpec.desc)
        self.assertEqual(self.data['data'], modelSpec.data)
        self.assertEqual(self.data['mbtrs'], modelSpec.mbtrs)
        self.assertEqual(self.data['krr'], modelSpec.krr)


    def test_init_from_yaml(self):
        ModelSpec.from_dict = MagicMock()

        modelSpec = ModelSpec.from_yaml(dirname + '/model_1.yaml')

        ModelSpec.from_dict.assert_called_with(self.data)


    def test_init_from_file(self):
        ModelSpec.from_dict = MagicMock()

        modelSpec = ModelSpec.from_file(dirname + '/model_1.spec.npy')

        ModelSpec.from_dict.assert_called_with(self.data)

    def test_defaults_for_mbtr(self):
        modelSpec = ModelSpec.from_yaml(dirname + '/model_mini.yaml')

        mbtr = modelSpec.mbtrs['mbtr_1']

        self.assertEqual(mbtr_defaults['distrf'], mbtr['distrf'])
        self.assertEqual(mbtr_defaults['corrf'], mbtr['corrf'])
        self.assertEqual(mbtr_defaults['norm'], mbtr['norm'])
        self.assertEqual(mbtr_defaults['flatten'], mbtr['flatten'])
        self.assertEqual(mbtr_defaults['elems'], mbtr['elems'])
        self.assertEqual(mbtr_defaults['acc'], mbtr['acc'])
        self.assertEqual(modelSpec.version, 0.1)
