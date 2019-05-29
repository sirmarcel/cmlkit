"""The backend to keep track of trials and their results."""
import numpy as np
from copy import deepcopy

from cmlkit.engine import parse_config


class ResultDB:
    """A database of key -> evaluation result mappings.

    Any backend that behaves roughly like a dict is fine,
    for instance a diskcache `Index` or a plain `dict`.

    Results are stored as key -> [state, outcome], i.e. the
    list version of the standard result format. I found it
    necessary to have state separated out to make searching
    by state slightly less tedious.

    """

    def __init__(self, db=None):
        super().__init__()

        # setting {} as default value can lead to unforeseen trouble,
        # so we use a sentinel value
        if db is None:
            db = {}

        self.db = db

    def __setitem__(self, key, value):
        self.submit(key, *value)

    def submit(self, key, state, outcome):
        outcome = deepcopy(outcome)

        self.db[key] = [state, outcome]

    def submit_result(self, key, result):
        state, outcome = parse_config(result)

        self.submit(key, state, outcome)

    def where_state(self, state):
        """Return only keys where state."""

        def is_state(x):
            return self[x][0] == state

        return list(filter(is_state, self.keys()))

    def get_outcome(self, key):
        return self[key][1]

    def get_result(self, key):
        state, outcome = self[key]
        return {state: outcome}

    def losses(self):
        """Keys, losses in order of insertion."""

        keys = self.where_state("ok")
        losses = [self.get_outcome(k)["loss"] for k in keys]

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
            error = self.get_outcome(k).get("error", "UnknownError")

            if error in errors:
                errors[error] += 1
            else:
                errors[error] = 1

        return errors

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
