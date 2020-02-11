from pickle import UnpicklingError

from cmlkit import logger
from cmlkit.engine.inout import read_npy, safe_save_npy

from .cache import Cache


class DiskCache(Cache):
    """Rudimentary disk cache.

    WARNING: NO CLEANUP IS DONE. BE CAREFUL!
    """

    def __init__(self, location):
        super().__init__()

        self.location = location

    def filename(self, key):
        return self.location / f"{key}.cache.npy"

    def check(self, key):
        return self.filename(key).is_file()

    def store(self, key, value):
        self.location.mkdir(parents=True, exist_ok=True)

        safe_save_npy(self.filename(key), {"value": value})

    def retrieve(self, key):
        return read_npy(self.filename(key))["value"]

    def try_retrieve(self, key):
        # sometimes, corrupted data is written to disk
        # this catches it and deletes the corrupted data
        try:
            return self.retrieve(key)
        except (EOFError, OSError, UnpicklingError):
            filename = self.filename(key)
            logger.error(f"Could not read cache file {filename}; deleting it.")
            filename.unlink()

            return None
