from .cache import Cache


class NoCache(Cache):
    """Dummy cache."""

    def check(self, key):
        return False

    def store(self, key, data):
        pass
