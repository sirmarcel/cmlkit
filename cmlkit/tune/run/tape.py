"""The tape on which all optimisation steps are recorded.

This is basically just a tiny wrapper to make sure we can
use either just a list (for "result mode") or something file-backed
for the tape.

"""

from cmlkit.engine.inout import read_son, save_son


class Tape:
    """Tape.

    The tape is the record of an optimisation. We employ it
    in three different modes:
    - *during* an optimisation, where steps are written to disk with append(),
    - when restarting a run, where we only need to read the tape back (it is re-written
      somewhere else),
    - when checking out the result of a run, where we want everything to be in memory.

    Under the hood, this boils down to two cases: either we have a file backend,
    or we do everything with a plain list.
    """

    def __init__(self, backend, tape, metadata):
        self.backend = backend
        self.tape = tape
        self.metadata = metadata

    @classmethod
    def new(cls, filename=None, metadata={}):
        if filename is None:
            metadata = metadata
            backend = "list"
            tape = []

        else:
            save_son(filename, metadata, is_metadata=True)
            metadata = metadata
            backend = "son"
            tape = filename

        return cls(backend, tape, metadata)

    @classmethod
    def restore(cls, filename):
        backend = "list"
        metadata, tape = read_son(filename)

        return cls(backend, tape, metadata)

    def append(self, item):
        if self.backend == "list":
            self.tape.append(item)
        else:
            save_son(self.tape, item)

    @property
    def raw(self):
        if self.backend == "list":
            return self.tape
        else:
            meta, data = read_son(self.tape)
            return data

    def __iter__(self):
        if self.backend == "list":
            return iter(self.tape)

        else:
            meta, data = read_son(self.tape)
            return iter(data)
