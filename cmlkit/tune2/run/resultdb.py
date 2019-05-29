"""The backend to keep track of trials and their results."""
import numpy as np
from copy import deepcopy


class ResultDB:
    """A database of key -> evaluation result mappings."""

    def __init__(self, db=None):
        super().__init__()

        # setting {} as default value can lead to unforeseen trouble,
        # so we use a sentinel value
        if db is None:
            db = {}

        self.db = db

    def __setitem__(self, key, value):
        self.submit(key, *value)

    def submit(self, key, state, inner):
        inner = deepcopy(inner)

        self.db[key] = [state, inner]

    def where_state(self, state):
        """Return only keys where state."""

        def is_state(x):
            return self[x][0] == state

        return list(filter(is_state, self.keys()))

    def get_result(self, key):
        return self[key][1]

    def losses(self):
        """Keys, losses in order of insertion."""

        keys = self.where_state("ok")
        losses = [self.get_result(k)["loss"] for k in keys]

        return keys, losses

    def sorted_losses(self):
        """Keys, losses in ascending order."""

        keys, losses = self.losses()
        sorted_losses = sorted(losses)

        sorted_keys = [keys[losses.index(l)] for l in sorted_losses]

        return sorted_keys, list(sorted_losses)

    def count_by_error(self):
        """Return a dict mapping error classes to counts"""

        errors = {}

        for k in self.where_state("error"):
            error = self.get_result(k).get("error", "UnknownError")

            if error in errors:
                errors[error] += 1
            else:
                errors[error] = 1

        return errors

    # probably not needed!

    def __getitem__(self, key):
        return self.db[key]

    def __delitem__(self, key):
        return self.db.__delitem__(key)

    def __iter__(self):
        return (self.db).__iter__()

    def __len__(self):
        return (self.db).__len__()

    def __contains__(self, key):
        return self.db.__contains__(key)

    def keys(self):
        return self.db.keys()

    def values(self):
        return self.db.values()

    def items(self):
        return self.db.items()
