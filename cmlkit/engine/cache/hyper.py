from pathlib import Path

from cmlkit.engine import parse_config, compute_hash

from .disk import DiskCache


class HyperCache:
    """Cache manager."""

    def __init__(self, location):
        self.location = Path(location)

        self.caches = []

    def register(self, component):
        cache_config = component.context.get("cache", None)  # default to off

        if cache_config:
            key = f"{component.kind}/{compute_hash(component.get_config())}"
            cache_kind, cache_inner = parse_config(cache_config, shortcut_ok=True)

            if cache_kind == "disk":
                cache = DiskCache(location=self.location / key)
                self.caches.append((str(component), key, cache))

            else:
                raise NotImplementedError(
                    f"Currently, only 'disk' type caches are supported. Not {cache_config}."
                )

            return cache

        else:
            return None
