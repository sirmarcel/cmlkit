from .cache import Cache


class NoCache(Cache):
    """Dummy cache."""

    def __init__(self, ):
        super().__init__()

    def check(self, key):
        return False
