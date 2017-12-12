from unittest import TestCase
from indices import *


class TestFourwaySplit(TestCase):

    def test_element_numbers_sum_up(self):
        n = 40
        k_test = 5
        k_valid = 15
        k_train = 7

        a, b, c, d = fourway_split(n, k_train, k_test, k_valid)

        self.assertEqual(len(a) + len(b) + len(c) + len(d), n)

    def test_union_is_all(self):
        n = 100
        k_test = 25
        k_valid = 15
        k_train = 36

        a, b, c, d = fourway_split(n, k_train, k_test, k_valid)

        union = np.union1d(a, b)
        union = np.union1d(union, c)
        union = np.union1d(union, d)

        self.assertEqual(union.all(), np.array(np.arange(n)).all())


class TestThreewaySplit(TestCase):

    def test_element_numbers_sum_up(self):
        n = 30
        k_test = 5
        k_valid = 15

        a, b, c = threeway_split(n, k_test, k_valid)

        self.assertEqual(len(a) + len(b) + len(c), n)

    def test_union_is_all(self):
        n = 100
        k_test = 25
        k_valid = 15

        a, b, c = threeway_split(n, k_test, k_valid)

        union = np.union1d(a, b)
        union = np.union1d(union, c)

        self.assertEqual(union.all(), np.array(np.arange(n)).all())


class TestGenerateDistinctSets(TestCase):

    def test_element_numbers_sum_up(self):
        n = 78
        full = np.arange(n)
        k = 64
        a, b = generate_distinct_sets(full, k)
        self.assertEqual(len(a) + len(b), n)

    def test_union_is_all(self):
        n = 78
        full = np.arange(n)
        k = 3
        a, b = generate_distinct_sets(full, k)
        self.assertEqual(np.union1d(a, b).all(), np.array(full).all())

    def test_set_disjunct(self):
        n = 78
        full = np.arange(n)
        k = 3
        a, b = generate_distinct_sets(full, k)
        self.assertTrue(np.intersect1d(a, b).size == 0)
