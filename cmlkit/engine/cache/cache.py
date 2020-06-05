class Cache:
    """Cache base class."""

    def __init__(self):
        self.misses = 0
        self.hits = 0

    def __contains__(self, key):
        """Is key in cache?"""

        found = self.check(key)
        if not found:
            self.misses += 1

        return found

    def get(self, key):
        """Get cached result."""

        if not self.check(key):
            raise KeyError("{key} not found in cache {self}.")

        self.hits += 1
        return self.retrieve(key)

    def get_if_cached(self, key):
        """Attempt to get cached result.

        Returns:
            Cached result if available, otherwise None.
        """

        if key not in self:
            return None

        data = self.try_retrieve(key)
        if data is not None:
            self.hits += 1

        return data

    def submit(self, key, data):
        # wrapping just in case
        self.store(key, data)

    def try_retrieve(self, key):
        # overload this to implement
        # "risky" retrieval that may
        # fail. if it fails, return None
        return self.retrieve(key)

    def check(self, key):
        # implement: is "key" in cache, bool
        ...

    def retrieve(self, key):
        # implement getting cached result
        ...

    def store(self, key, data):
        # implement storing result to be cached
        ...
