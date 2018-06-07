from unittest import TestCase
from cmlkit.indices import *


class TestFourwaySplit(TestCase):
    def setUp(self):
        self.n = 40
        self.k_train = 7
        self.k_test = 5
        self.k_valid = 15

        self.a, self.b, self.c, self.d = fourway_split(self.n, self.k_train, self.k_test, self.k_valid)

    def test_sizes(self):
        self.assertEqual([len(self.a), len(self.b), len(self.c), len(self.d)], [40-5-15-7, 7, 5, 15])

    def test_union_is_all(self):
        union = np.union1d(self.a, self.b)
        union = np.union1d(union, self.c)
        union = np.union1d(union, self.d)

        self.assertEqual(union.all(), np.array(np.arange(self.n)).all())



class TestTwowaySplit(TestCase):
    def setUp(self):
        self.n = 40
        self.k_test = 10

        self.a, self.b = twoway_split(self.n, self.k_test)

    def test_sizes(self):
        self.assertEqual([len(self.a), len(self.b)], [40-10, 10])

    def test_union_is_all(self):
        union = np.union1d(self.a, self.b)

        self.assertEqual(union.all(), np.array(np.arange(self.n)).all())



class TestThreewaySplit(TestCase):
    def setUp(self):
        self.n = 40
        self.k_test = 5
        self.k_valid = 15

        self.a, self.b, self.c = threeway_split(self.n, self.k_test, self.k_valid)

    def test_sizes(self):
        self.assertEqual([len(self.a), len(self.b), len(self.c)], [40-5-15, 5, 15])

    def test_union_is_all(self):
        union = np.union1d(self.a, self.b)
        union = np.union1d(union, self.c)

        self.assertEqual(union.all(), np.array(np.arange(self.n)).all())



class TestGenerateIndices(TestCase):

    def test_make_range_if_int(self):
        ind = generate_indices(6, [])
        self.assertEqual(ind.all(), np.arange(6).all())

    def test_pass_through_index_array(self):
        ind = generate_indices(np.arange(6), [])
        self.assertEqual(ind.all(), np.arange(6).all())

    def test_exclude(self):
        ind = generate_indices(6, [3])
        self.assertFalse(3 in ind)



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
