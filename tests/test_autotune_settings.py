from unittest import TestCase
from unittest.mock import MagicMock
import hyperopt as hp
from qmmltools.autotune.settings import *
import qmmltools.stats as qmts


class TestHyperoptConversions(TestCase):
    def setUp(self):
        self.og_choice = hp.choice

    def tearDown(self):
        hp.choice = self.og_choice

    def test_to_hyperopt(self):
        hp.choice = MagicMock()
        r = to_hyperopt(('hp_choice', 'mbtr_1', [1, 2, 3]))
        hp.choice.assert_called_with('mbtr_1', [1, 2, 3])

    def test_is_hyperopt(self):

        self.assertTrue(is_hyperopt(('hp_choice', 'mbtr_1', [1, 2, 3])))


class TestParsing(TestCase):

    def setUp(self):
        self.og_choice = hp.choice
        self.og_rmse = qmts.rmse

    def tearDown(self):
        hp.choice = self.og_choice
        qmts.rmse = self.og_rmse

    def test_basic(self):
        d = {
            'one': {'mbtr1': ('hp_choice', 'mbtr_1', [1, 2, 3]),
                    'loss': 'rmse',
                    'other': 'something_else'},
            'two': 'heh'

        }

        hp.choice = MagicMock()
        qmts.rmse = 'hey_it_worked'

        parse_settings(d)

        hp.choice.assert_called_with('mbtr_1', [1, 2, 3])
        self.assertEqual(d['one']['loss'], 'hey_it_worked')
