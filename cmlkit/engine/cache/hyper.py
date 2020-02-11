from pathlib import Path

from cmlkit.engine import parse_config, compute_hash

from .disk import DiskCache
from .no import NoCache


class HyperCache:
    """Cache manager."""

    def __init__(self, location):
        self.location = Path(location)

        self.caches = []

    def register(self, component):
        cache_config = component.context.get("cache", "no")  # default to off

        key = f"{component.kind}/{compute_hash(component.get_config())}"
        cache_kind, cache_inner = parse_config(cache_config, shortcut_ok=True)

        if cache_kind == "disk":
            cache = DiskCache(location=self.location / key)
            self.caches.append((str(component), key, cache))

        elif cache_kind == "no":
            cache = NoCache()

        else:
            raise NotImplementedError(
                "Currently, only 'disk' and 'no' type caches are supported."
            )

        return cache
