from unittest import TestCase
from cmlkit.helpers import *


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


class TestListsToTuples(TestCase):

    def test_lists_to_tuples(self):
        d = {
            'a': '42',
            'b': [1, 2, 3, [42, 1, 3]],
            'f': {'m': ['lgs']},
            'c': {'m': ['lgs', [1, 2, 3]]},
            'e': {'m': ['lgs', '42']}
        }

        lists_to_tuples(d)
        print(d)

        self.assertIsInstance(get_with_path(d, ['b']), tuple)
        self.assertIsInstance(get_with_path(d, ['b', 3]), tuple)
        self.assertIsInstance(get_with_path(d, ['f', 'm']), tuple)
        self.assertIsInstance(get_with_path(d, ['c', 'm']), tuple)
        self.assertIsInstance(get_with_path(d, ['e', 'm']), tuple)


class TestTuplesToLists(TestCase):

    def test_tuples_to_lists(self):
        d = {
            'a': '42',
            'b': [1, 2, 3, (42, 1, 2, 3)],
            'f': {'m': ['lgs', [1, 2, 3, '42']]},
            'c': {'m': ('lgs', [1, 2, 3])},
            'e': {'m': ['lgs', '42']},
            'tricky': (1, 2, [1, 2, (3, 4)])
        }

        tuples_to_lists(d)
        print(d)

        self.assertIsInstance(get_with_path(d, ['b']), list)
        self.assertIsInstance(get_with_path(d, ['b', 3]), list)
        self.assertIsInstance(get_with_path(d, ['f', 'm']), list)
        self.assertIsInstance(get_with_path(d, ['f', 'm', 1]), list)
        self.assertIsInstance(get_with_path(d, ['c', 'm']), list)
        self.assertIsInstance(get_with_path(d, ['e', 'm']), list)
        self.assertIsInstance(get_with_path(d, ['tricky']), list)
        self.assertIsInstance(get_with_path(d, ['tricky', 2]), list)
        self.assertIsInstance(get_with_path(d, ['tricky', 2, 2]), list)



class TestPathHelpers(TestCase):

    def test_get_path(self):
        d = {
            'a': '42',
            'b': [1, 2, 3, [['42']]],
            'f': {'m': ['lgs', [1, 2, 3, '42']]},
            'c': {'m': ['lgs', [1, 2, 3]]},
            'e': {'m': ['lgs', '42']}
        }

        self.assertEqual(get_with_path(d, ['f', 'm', 0]), 'lgs')

    def test_set_path(self):
        d = {
            'a': '42',
            'b': [1, 2, 3, [['42']]],
            'f': {'m': ['lgs', [1, 2, 3, '42']]},
            'c': {'m': ['lgs', [1, 2, 3]]},
            'e': {'m': ['lgs', '42']}
        }

        set_with_path(d, ['f', 'm', 0], '1234')

        self.assertEqual(get_with_path(d, ['f', 'm', 0]), '1234')

    def test_find_pattern(self):

        d = {
            'a': '42',
            'b': [1, 2, 3, [['42']]],
            'f': {'m': ['lgs', [1, 2, 3, '42']], 'c': ['42']},
            'c': {'m': ['lgs', [1, 2, 3]]},
            'e': {'m': ['lgs', '42']}
        }

        def my_pattern(value):
            return value == '42'

        found = find_pattern(d, my_pattern)

        for f in found:
            self.assertTrue(my_pattern(get_with_path(d, f)))

        self.assertEqual(len(found), 5)

    def test_find_pattern_ignore(self):
        d = {
            'a': '42',
            'b': [1, 2, 3, [['42']]],
            'f': {'m': ['lgs', [1, 2, 3, '42']]},
            'c': {'m': ['lgs', [1, 2, 3]]},
            'e': {'m': ['lgs', '42', '42']}
        }

        def my_pattern(value):
            return value == '42'

        found = find_pattern(d, my_pattern, ignore=[['f', 'm']])

        self.assertNotEqual(found, [['a'], ['b', 3, 0, 0], ['f', 'm', 1, 3], ['e', 'm', 1], ['e', 'm', 2]])
        self.assertEqual(found, [['a'], ['b', 3, 0, 0], ['e', 'm', 1], ['e', 'm', 2]])

    def test_match_paths(self):
        ignore = [['f', 'm'], ['b']]

        self.assertTrue(matches_any(['f', 'm'], ignore))
        self.assertTrue(matches_any(['b'], ignore))
        self.assertFalse(matches_any(['c'], ignore))
