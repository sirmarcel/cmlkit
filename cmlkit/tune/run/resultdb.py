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

    # submission
    # using set is discouraged

    def __setitem__(self, key, value):
        self.submit(key, *value)

    def submit(self, key, state, outcome):
        outcome = deepcopy(outcome)

        self.db[key] = [state, outcome]

    def submit_result(self, key, result):
        state, outcome = parse_config(result)

        self.submit(key, state, outcome)

    # user interface

    def where_state(self, state):
        """Return only keys whose state field matches state."""

        def is_state(x):
            return self[x][0] == state

        return list(filter(is_state, self.keys()))

    def get_outcome(self, key):
        """Return the naked result, not including the status."""
        return self[key][1]

    def get_result(self, key):
        """Return a standard result dict, {"status": {#outcome}}."""
        state, outcome = self[key]
        return {state: outcome}

    def losses(self):
        """Losses in order of insertion."""

        keys = self.where_state("ok")
        return [self.get_outcome(k)["loss"] for k in keys]

    def sorted_losses(self):
        """Losses sorted ascending."""

        return list(sorted(self.losses()))

    def tids_losses(self):
        """Keys, losses in order of insertion."""

        keys = self.where_state("ok")
        losses = [self.get_outcome(k)["loss"] for k in keys]

        return keys, losses

    def sorted_tids_losses(self):
        """Keys, losses in sorted ascending."""

        keys, losses = self.tids_losses()
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

    def top_suggestions(self, n=5):
        """Return n suggestions sorted by loss."""
        tids, losses = self.sorted_tids_losses()

        return [self.get_outcome(tids[i])["suggestion"] for i in range(n)]

    def top_refined_suggestions(self, n=5):
        """Return n refined suggestions sorted by loss, or {} if not available."""
        tids, losses = self.sorted_tids_losses()

        return [self.get_outcome(tids[i]).get("refined_suggestion", {}) for i in range(n)]

    def top_losses(self, n=5):
        """Return top n losses."""
        losses = self.sorted_losses()

        if len(losses) == 0:
            return []
        else:
            # so we don't get an error if we request this
            # at an early run stage
            return losses[0:min(n, len(losses))]

    # housekeeping; forward various magic methods to the underlying db

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
