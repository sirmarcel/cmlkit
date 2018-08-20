import numpy as np
import hashlib


def hash_sortable_dict(d):
    """Compute a stable non-cryptographic hash of a dict with sortable keys.

    In more concrete terms, d is expected to be a dictionary with only strings as keys, and lists,
    tuples, ndarrays, strings, numbers, or other dicts of a similar form as dicts.

    Args:
        d: A dict where the keys can be sorted (strings, ints, etc.)

    Returns:
        hash: A string with the hexadecimal representation of the hash

    """

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


def hash_arrays(*args):
    """Compute a stable non-cryptographic hash of a number of ndarrays

    NOTE: This hash is NOT invariant under partitioning of the arrays, i.e.
    [[1, 2], [3, 4]] hashes to the same as [1, 2, 3, 4]. I'm not sure why
    this happens, but I don't think it'll be a problem at present.

    Args:
        *args: A number of ndarrays

    Returns:
        hash: A string with the hexadecimal representation of the hash

    """

    f = hashlib.md5()
    for a in args:
        _hash(a, f)

    return f.hexdigest()
