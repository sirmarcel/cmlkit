"""Caching functions.

Implements in-memory and on-disk caching. Both are
quite rudimentary at the moment, and will need some
kind of overhaul at a future point. A particular
sticking point is being able to control the memory/disk
usage, and mabye dynamically switch between disk and memory.

It might also be useful to be more restrictive in the kind
of data that can be stored/hashed.

Caveats:
- The current hash function is quite slow (especially for large arrays)
- Storing mutable objects in the mem cache can lead to unintended consequences
- There is absolutely no monitoring of disk/mem usage

The original code for this was written by Matthias Rupp.

"""

import sys
import os
import glob
import time
import hashlib
import pickle
import numpy as np

from cmlkit.engine import read_npy, safe_save_npy, makedir
from cmlkit.engine.hashing import compute_hash
from cmlkit import logger


def make_memcached(max_entries=500):
    """Decorator to cache a function in memory.

    A maximum number of results to be stored can be specified via max_entries."""

    return lambda f: memcached(f, max_entries=max_entries)


class memcached:
    """Implementation of memoization decorator."""

    def __init__(self, f, max_entries=500):
        """Wraps f in a caching wrapper."""
        self.f = f  # cached function
        self.max_entries = max_entries  # maximum number of results to store
        self.cache = {}  # stores pairs of hash keys and function results
        self.lruc = 0  # least recently used counter
        self.hashf = None  # hashing function, initialized by __call__
        self.hits, self.misses = (
            0,
            0,
        )  # number of times function values were retrieved from cache/had to be computed
        self.total_hits, self.total_misses = (
            0,
            0,
        )  # statistics over lifetime of cached object

    def __call__(self, *args, **kwargs):
        """Calls to cached function."""

        key = compute_hash(*args, **kwargs)

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


def make_discached(cache_location, name=""):
    """Decorator to cache a function on disk."""

    return lambda f: diskcached(f, cache_location)


class diskcached:
    """Cache the function to disk in a specified location.

    Note that hashes are somewhat fickle to make consistent across Python
    restarts -- you should probably test this for your usecase before
    committing large amounts of cpu power.
    """

    def __init__(self, f, cache_location, name="noname", min_duration=0.5):
        """Wraps f in a caching wrapper."""
        self.f = f  # cached function
        self.cache_location = (
            os.path.normpath(cache_location) + "/"
        )  # location of cache on disk
        self.name = name  # unique name of function to be cached, is hashed w/ args
        self.min_duration = (
            min_duration
        )  # min duration to bother caching (to avoid caching pointless things)
        self.hits, self.misses = (
            0,
            0,
        )  # number of times function values were retrieved from cache/had to be computed

    def __call__(self, *args, **kwargs):
        """Calls to cached function."""

        key = compute_hash(self.name, *args, **kwargs)

        filename = self.cache_location + self.name + "." + key + ".cache.npy"

        if os.path.isfile(filename):
            try:
                data = read_npy(filename)
                self.hits += 1
                return data["val"]
            except (EOFError, OSError, pickle.UnpicklingError):
                # All of these occur when the cached file is corrupted.

                logger.error(
                    f"Could not read cache file {filename}; deleting and recomputing."
                )
                os.remove(filename)
                return self.compute_cache_result(args, kwargs, filename)

        else:
            return self.compute_cache_result(args, kwargs, filename)

    def compute_cache_result(self, args, kwargs, filename):
        start = time.time()
        val = self.f(*args, **kwargs)
        stop = time.time()
        duration = stop - start

        if duration > self.min_duration:
            tosave = {"val": val, "name": self.name, "duration": duration}
            makedir(self.cache_location)  # make sure cache_location exists
            safe_save_npy(filename, tosave)

        self.misses += 1
        return val
