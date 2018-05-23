import numpy as np
import hashlib

def hash_spec_dict(d):
    """Compute a stable noncryptographic hash of a dict specifying a model or something similar.

    In more concrete terms, d is expected to be a dictionary with only strings as keys, and lists,
    tuples, ndarrays, strings, numbers, or other dicts of a similar form as dicts."""

    f = hashlib.md5()
    for v in sorted(d.items()):
        _hash(v, f)

    return f.hexdigest()


def _hash(item, f):
    if isinstance(item, np.ndarray) and item.dtype == object:
        for i in item:
            _hash(i, f)
    elif item is None:
        _hash('None', f)
    elif isinstance(item, dict):
        for v in sorted(item.items()):
            _hash(v, f)
    elif isinstance(item, (tuple, list)):
        for i in item:
            _hash(i, f)
    elif isinstance(item, str):
        f.update(item.encode('utf-8'))
    elif isinstance(item, float) or isinstance(item, int):
        _hash(str(item), f)
    else:
        try:
            f.update(item)
        except:
            raise Exception("No known unique hash for " + str(item))
