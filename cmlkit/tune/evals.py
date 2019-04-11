import os
from copy import deepcopy
from cmlkit2.engine import safe_save_npy, Configurable


class Evals(Configurable):
    """Convenience wrapper for id -> Evaluator result mappings"""

    kind = 'evals'

    def __init__(self, db={}, name='evals', normalise=False, context={}):
        super().__init__()

        self.db = db
        self.name = name

        if normalise:
            self.normalise()

    def _get_config(self):
        return {'db': self.db, 'name': self.name}

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def save(self, dirname='', filename=None):
        """Save to disk, defaulting to the name as filename"""

        if filename is None:
            filename = self.name

        # save in separate thread by default, for sanity
        safe_save_npy(os.path.join(dirname, filename), self.get_config())

    def __setitem__(self, key, val):
        self.submit(key, val)

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

    def normalise(self):
        for k, v in self.items():
            self.submit(k, v)

    def submit(self, key, result):
        result = deepcopy(result)

        # TODO: centralise this default behaviour somewhere; it is duplicated from SearchBase
        # Set defaults: status is ok, default error type is Exception
        result['status'] = result.get('status', 'ok')

        if result['status'] == 'error':
            result['error'] = result.get('error', (Exception.__class__.__name__, str(Exception)))

        self.db[key] = result

    def by_key(self, key):
        """Return self.vals() sorted by key"""

        sorted_keys = self.ids_by_key(key)
        return [self[k] for k in sorted_keys]

    def where(self, key, val):
        """Return self.vals() where vals[key] == val"""

        results = []
        for k, v in self.db.items():
            compare = v.get(key, 'not_found')  # hacky

            if compare == val:
                results.append(v)

        return results

    def ids_by_key(self, key):
        """Return self.keys() sorted by key"""
        def get_key(x):
            # if the key is not found, take it to be infinity,
            # this allows us to deal with results that ended
            # in errors, where no losses are written
            return self[x].get(key, float('inf'))

        sorted_keys = sorted(self.keys(), key=lambda x: get_key(x))
        return sorted_keys

    def count_by_error(self):
        """Return a dict mapping error classes to counts"""

        errors = {}

        for v in self.values():
            if v['status'] == 'error':
                error = v['error']

                if error in errors:
                    errors[error] += 1
                else:
                    errors[error] = 1

        return errors
