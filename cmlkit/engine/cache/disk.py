from pickle import UnpicklingError

from cmlkit import logger
from cmlkit.engine.data import load_data

from .cache import Cache


class DiskCache(Cache):
    """Rudimentary disk cache.

    WARNING: NO CLEANUP IS DONE. BE CAREFUL!

    Defaults to dumping protocol 1, which is
    non-compressed .npz files. If you know you
    will be generating easily compressible files
    you can manually overwrite this in the component
    context.
    """

    def __init__(self, location, protocol=1):
        super().__init__()

        self.location = location
        self.protocol = protocol

    def filename(self, key):
        return self.location / (key + ".npz")

    def check(self, key):
        return self.filename(key).is_file()

    def store(self, key, data):
        self.location.mkdir(parents=True, exist_ok=True)
        data.dump(self.filename(key), protocol=self.protocol)

    def retrieve(self, key):
        return load_data(self.filename(key))

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
