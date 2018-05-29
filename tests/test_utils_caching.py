from unittest import TestCase
from unittest.mock import MagicMock, patch
from qmmltools.utils.caching import _memcached, _diskcached, stable_hash
import hashlib
import numpy as np
b = np.array(['a', 'b'], dtype=object)


class TestMemCached(TestCase):

    def test_caching_works(self):
        with MagicMock() as f:
            f2 = _memcached(f, 2)
            f2(1)
            f2(1)

            f.assert_called_once_with(1)


class TestDiskCached(TestCase):

    @patch('qmmltools.inout.save')
    @patch('qmmltools.inout.read')
    def test_uses_cache(self, mock_read, mock_save):
        mock_read.return_value = {'val': 2}

        with MagicMock() as f:
            f.return_value = 2
            f2 = _diskcached(f, '')
            f2(1)
            f2(1)

            f.assert_not_called()


    @patch('qmmltools.inout.save')
    def test_calls_save(self, mock_save):
        # mock_read.return_value = {'val': 2, 'args': [1]}

        def f(x):
            return x*2

        f2 = _diskcached(f, '')
        f2(1)

        mock_save.assert_called_once_with('./c4ca4238a0b923820dcc509a6f75849b.cache', {'val': 2, 'name': ''})


class TestHashStability(TestCase):

    def test_hashes_object_arrays(self):
        a = np.array(['a', 'b'], dtype=object)
        hash1 = stable_hash(a)
        # This hash was computed at some previous run -- it should always be the same!
        self.assertEqual(hash1, '187ef4436122d1cc2f40dc2b92f0eba0')

    def test_hashes_object_arrays_invariant_under_permutation(self):
        a = np.array(['a', 'b'], dtype=object)
        a = np.array(['b', 'a'], dtype=object)
        hash1 = stable_hash(a)
        hash2 = stable_hash(a)
        self.assertEqual(hash1, hash2)

    def test_hashes_string(self):
        a = 'hello'
        hash1 = stable_hash(a)
        print(hash1)
        # This hash was computed at some previous run -- it should always be the same!
        self.assertEqual(hash1, '5d41402abc4b2a76b9719d911017c592')
