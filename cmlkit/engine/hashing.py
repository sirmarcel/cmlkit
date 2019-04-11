import numpy as np
import hashlib
import joblib
"""Implements hashes that are (hopefully) stable across sessions."""
# Note: I have found a hash instability in my implementation below,
# which produces different hashes for dicts that were deepcopied
# from each other... which is VERY problematic. The hashes are stable
# across restarts, which is doubly strange.

# Will use the joblib implementation for cases like this,
# which seems to not suffer from this problem. Joblibs hashes
# seem to be much slower (I suspect the dumping of the ndarrays is problematic)
# so I will keep both for now...


def compute_hash(*args, **kwargs):
    to_hash = {'args': args, 'kwargs': kwargs}
    return joblib.hash(to_hash)


fast_hash = compute_hash  # I am fed up with fixing this shit

# def fast_hash(*items, **kwitems):
#     """Return hexdigest of argument(s).

#     Caveats:
#         Hash is NOT invariant under partitioning of ndarrays, i.e.
#         [[1, 2], [3, 4]] hashes to the same as [1, 2, 3, 4].
#         TODO: Maybe fix this by switching to a different hashing scheme
#         for numpy objects
#     """
#     hashf = hashlib.md5()

#     for i in items:
#         _hash(i, hashf)
#     for k, v in kwitems.items():
#         _hash(k, hashf)
#         _hash(v, hashf)

#     return hashf.hexdigest()


# def _hash(item, f):
#     if isinstance(item, np.ndarray) and item.dtype == object:
#         for i in item:
#             _hash(i, f)
#     elif item is None:
#         _hash('None', f)
#     elif isinstance(item, dict):
#         for v in sorted(item.items()):
#             _hash(v, f)
#     elif isinstance(item, (tuple, list)):
#         for i in item:
#             _hash(i, f)
#     elif isinstance(item, str):
#         f.update(item.encode('utf-8'))
#     elif isinstance(item, float) or isinstance(item, int) or isinstance(item, bool):
#         _hash(str(item), f)
#     else:
#         try:
#             f.update(hash(item))
#         except:
#             raise ValueError("Cannot hash " + str(item))
