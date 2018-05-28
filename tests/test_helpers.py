from unittest import TestCase
from helpers import *


class TestSequenceHandling(TestCase):

    def test_handles_single_values_in_sequence(self):
        a = ('gaussian')
        b = convert_sequence(a)

        self.assertEqual(b, 'gaussian')

    def test_handles_single_values_alone(self):
        a = 'gaussian'
        b = convert_sequence(a)

        self.assertEqual(b, 'gaussian')

    def test_handles_associated_values(self):
        a = ('gaussian', 1, 2, 3)
        b = convert_sequence(a)

        self.assertEqual(b, ('gaussian', (1, 2, 3)))

    def test_handles_mixed(self):
        a = ['normal', 16.1]
        b = convert_sequence(a)

        self.assertEqual(b, ('normal', (16.1,)))


class TestFindKeyApplyF(TestCase):

    def test_basic(self):
        d = {'arg1': 3,
             'arg2': 2}

        find_key_apply_f(d, 'arg2', lambda x: x**2)
        self.assertEqual(d, {'arg1': 3, 'arg2': 4})

    def test_handles_nesting(self):
        d = {'arg1': 3,
             'arg2': {'arg3': 5, 'arg4': 9}}

        find_key_apply_f(d, 'arg4', lambda x: x**2)
        self.assertEqual(d, {'arg1': 3, 'arg2': {'arg3': 5, 'arg4': 9**2}})


class TestFindPatternApplyF(TestCase):

    def test_does_nothing_if_not_required(self):
        d = {'arg1': 3,
             'arg2': 2}

        def pattern(x):
            return isinstance(x, (tuple, list)) and x[0] == 't'

        find_pattern_apply_f(d, pattern, lambda x: x**2)
        self.assertEqual(d, {'arg1': 3, 'arg2': 2})

    def test_basic(self):
        d = {'arg1': 3,
             'arg2': ('t', 2)}

        def pattern(x):
            return isinstance(x, (tuple, list)) and x[0] == 't'

        find_pattern_apply_f(d, pattern, lambda x: x[1]**2)
        self.assertEqual(d, {'arg1': 3, 'arg2': 4})

    def test_handles_nesting(self):
        d = {'arg1': 3,
             'arg2': {'k': ('t', 2)}}

        def pattern(x):
            return isinstance(x, (tuple, list)) and x[0] == 't'

        find_pattern_apply_f(d, pattern, lambda x: x[1]**2)
        self.assertEqual(d, {'arg1': 3, 'arg2': {'k': 4}})

    def test_finds_in_seqeuence(self):
        d = {'arg1': 3,
             'arg2': ['k', ('t', 2)]}

        def pattern(x):
            return isinstance(x, (tuple, list)) and x[0] == 't'

        find_pattern_apply_f(d, pattern, lambda x: x[1]**2)
        self.assertEqual(d, {'arg1': 3, 'arg2': ['k', 4]})
