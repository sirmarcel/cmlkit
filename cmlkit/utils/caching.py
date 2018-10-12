import sys
import os
import hashlib
import numpy as np
import cmlkit.inout as cmlio


def memcached(max_entries=500):
    """Decorator to cache a function in memory.

    A maximum number of results to be stored can be specified via max_entries.

    functools.lru_cache does not accept mutable objects such as NumPy's ndarray."""

    return lambda f: _memcached(f, max_entries=max_entries)


class _memcached:
    """Implementation of memoization decorator.

    Function arguments need to be hashable, so not dictionaries and sequences instead of lists.
    """

    def __init__(self, f, max_entries=500):
        """Wraps f in a caching wrapper."""
        self.f = f  # cached function
        self.max_entries = max_entries  # maximum number of results to store
        self.cache = {}  # stores pairs of hash keys and function results
        self.lruc = 0  # least recently used counter
        self.hashf = None  # hashing function, initialized by __call__
        # self._intbuf = np.empty((1,), dtype=np.int)  # buffer for temporary storage of hash() values
        self.hits, self.misses = 0, 0  # number of times function values were retrieved from cache/had to be computed
        self.total_hits, self.total_misses = 0, 0  # statistics over lifetime of cached object

    def _hash(self, arg, hashf):
        """Hashes an argument. Note that these hashes are not consistent across restarts."""
        try:
            hashf.update(arg)
        except:
            # hash converts to something hashable,
            # and the np array supports the buffer API
            intbuf = np.empty((1,), dtype=np.int)  # buffer for temporary storage of hash() values
            intbuf[0] = hash(arg)
            hashf.update(intbuf)

    def __call__(self, *args, **kwargs):
        """Calls to cached function."""

        # compute hash key of arguments
        self.hashf = hashlib.md5()  # non-cryptographic fast hash function
        for i in args:
            self._hash(i, self.hashf)
        for i in kwargs:
            self._hash(kwargs[i], self.hashf)
        key = self.hashf.hexdigest()  # hexdigest for readability

        # return stored or computed value
        if key in self.cache:
            self.total_hits, self.hits = self.total_hits + 1, self.hits + 1
            item = self.cache[key]
            item[0] = self.lruc
            self.lruc += 1
            return item[1]
        else:
            self.total_misses, self.misses = self.total_misses + 1, self.misses + 1
            value = self.f(*args, **kwargs)
            self.cache[key] = [self.lruc, value]
            self.lruc += 1

            # remove another entry if new entry exceeded the maximum
            if len(self.cache) > self.max_entries:
                c, k = self.lruc, key
                for k_, v_ in self.cache.items():
                    if v_[0] < c:
                        c, k = v_[0], k_
                del self.cache[k]

            return value

    def clear_cache(self):
        """Empties cache, releasing memory."""
        self.cache.clear()
        self.hits, self.misses = 0, 0


def diskcached(cache_location, name=''):
    """Decorator to cache a function on disk.

    functools.lru_cache does not accept mutable objects such as NumPy's ndarray."""

    return lambda f: _diskcached(f, cache_location)


class _diskcached:
    """Cache the function to disk in a specified location.

    Note that hashes are somewhat fickle to make consistent across Python
    restarts -- you should probably test this for your usecase before
    committing large amounts of cpu power.
    """

    def __init__(self, f, cache_location, name=''):
        """Wraps f in a caching wrapper."""
        self.f = f  # cached function
        self.cache_location = os.path.normpath(cache_location) + '/'  # location of cache on disk
        self.name = name  # unique name of function to be cached, is hashed w/ args
        self.hashf = None  # hashing function, initialized by __call__
        self.hits, self.misses = 0, 0  # number of times function values were retrieved from cache/had to be computed

    def _hash(self, arg, hashf):
        """Hashes an argument."""

        if isinstance(arg, np.ndarray) and arg.dtype == object:
            for i in arg:
                self._hash(i, hashf)
        elif arg is None:
            self._hash('None', hashf)
        elif isinstance(arg, tuple):
            for i in arg:
                self._hash(i, hashf)
        elif isinstance(arg, str):
            hashf.update(arg.encode('utf-8'))
        elif isinstance(arg, float) or isinstance(arg, int):
            self._hash(str(arg), hashf)
        else:
            try:
                hashf.update(arg)
            except:
                # self._intbuf[0] = hash(arg)
                # self.hashf.update(self._intbuf)
                raise Exception("No known unique hash for " + str(arg))

    def __call__(self, *args, **kwargs):
        """Calls to cached function."""

        # compute hash key of arguments
        self.hashf = hashlib.md5()  # non-cryptographic fast hash function
        self._hash(self.name, self.hashf)
        for i in args:
            self._hash(i, self.hashf)
        for i in kwargs:
            self._hash(kwargs[i], self.hashf)

        key = self.hashf.hexdigest()  # hexdigest for readability

        # return stored or computed value
        try:
            data = cmlio.read(self.cache_location + key + '.cache')
            self.hits += 1
            return data['val']
        except FileNotFoundError:
            val = self.f(*args, **kwargs)
            tosave = {'val': val, 'name': self.name}
            cmlio.makedir(self.cache_location)  # make sure cache_location exists
            cmlio.save(self.cache_location + key + '.cache', tosave)
            self.misses += 1
            return val
        else:
            raise Exception("Something went wrong while accessing the disk cache at " + self.cache_location)


def stable_hash(x):
    # This function should always reflect they
    # way hashing is done for memcached, for testing
    def f(y):
        return y
    dummy = _diskcached(f, '')
    hashf = hashlib.md5()
    dummy._hash(x, hashf)
    return hashf.hexdigest()
