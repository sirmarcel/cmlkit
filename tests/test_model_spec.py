from unittest import TestCase
from unittest.mock import MagicMock
import qmmltools.inout as qmtio
import os
from qmmltools.model_spec import *
from bson.son import SON

dirname = os.path.dirname(os.path.abspath(__file__))


class TestModel(TestCase):
    def setUp(self):
        # load test data
        self.data = qmtio.read_yaml(dirname + '/model_1.spec.yml')

    def test_init_from_dict(self):
        modelSpec = ModelSpec.from_dict(self.data)

        self.assertEqual(self.data['name'], modelSpec.name)
        self.assertEqual(self.data['desc'], modelSpec.desc)
        self.assertEqual(self.data['data'], modelSpec.data)
        self.assertEqual(self.data['mbtrs'], modelSpec.mbtrs)
        self.assertEqual(self.data['krr'], modelSpec.krr)

    def test_init_from_yaml(self):
        ModelSpec.from_dict = MagicMock()

        modelSpec = ModelSpec.from_yaml(dirname + '/model_1.spec.yml')

        ModelSpec.from_dict.assert_called_with(self.data)

    def test_defaults_for_mbtr(self):
        modelSpec = ModelSpec.from_yaml(dirname + '/model_mini.spec.yml')

        mbtr = modelSpec.mbtrs['mbtr_1']

        self.assertEqual(mbtr_defaults['corrf'], mbtr['corrf'])
        self.assertEqual(mbtr_defaults['norm'], mbtr['norm'])
        self.assertEqual(mbtr_defaults['flatten'], mbtr['flatten'])
        self.assertEqual(mbtr_defaults['elems'], mbtr['elems'])
        self.assertEqual(mbtr_defaults['acc'], mbtr['acc'])


class TestConversions(TestCase):

    def test_convert_from_SON_if_subdict(self):
        d = {
            'a': 1,
            'b': SON({'l': 1})
        }

        d = convert_from_SON(d)

        self.assertFalse(isinstance(d['b'], SON))

    def test_convert_from_SON(self):
        d = SON({
            'a': 1,
            'b': SON({'l': 1})
        })

        d = convert_from_SON(d)

        self.assertFalse(isinstance(d['b'], SON))
        self.assertFalse(isinstance(d, SON))

    def test_convert_to_tuples(self):
        d = {
            'a': 1,
            'b': [1, 2, 3]
        }

        convert_to_tuple(d)

        self.assertTrue(isinstance(d['b'], tuple))
