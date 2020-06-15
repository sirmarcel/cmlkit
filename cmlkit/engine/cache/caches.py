from pathlib import Path

from cmlkit.engine import parse_config, compute_hash

from .disk import DiskCache
from .no import NoCache


class Caches:
    """Cache manager."""

    def __init__(self, location):
        self.location = Path(location)

        self.caches = []

    def register(self, component):
        # in case it gets overwritten!
        self.location = Path(self.location)

        cache_config = component.context.get("cache", "no")  # default to off

        if cache_config:
            key = f"{component.kind}/{component.get_hash()}"
            cache_kind, cache_inner = parse_config(
                cache_config, shortcut_ok=True
            )

            if cache_kind == "no":
                return NoCache()

            elif cache_kind == "disk":
                if "location" in cache_inner:
                    cache_inner["location"] = Path(cache_inner["location"]) / key
                else:
                    cache_inner["location"] = self.location / key

                cache = DiskCache(**cache_inner)
                self.caches.append((str(component), key, cache))

            else:
                raise NotImplementedError(
                    f"Currently, only 'disk' type caches are supported. Not {cache_config}."
                )

            return cache

        else:
            return None
