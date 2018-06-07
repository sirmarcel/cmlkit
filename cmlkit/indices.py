import numpy as np


def fourway_split(n, k_train, k_validate, k_test, exclude=[]):
    """Split the total n indices into four subsets.

    Items are picked at random and not repeated.

    Args:
        n: Total number of indices, OR ndarray of ints
        k_train: Number of items in the first set
        k_validate: Number of items in the second set
        k_test: Number of items in the third set
        exclude: optional, list of indices to exclude

    Returns:
        rest: Fourth set with remaining items
        train: First set (k_train items)
        validate: Second set (k_validate items)
        test: Third set (k_test items)
    """

    full = generate_indices(n, exclude)

    rest, train = generate_distinct_sets(full, k_train)
    rest, validate = generate_distinct_sets(rest, k_validate)
    rest, test = generate_distinct_sets(rest, k_test)

    return rest, train, validate, test



def threeway_split(n, k_validate, k_test, exclude=[]):
    """Generate training, validation and training index sets for n items.

    Items are picked at random and not repeated.

    In this split, the remaining items are returned first.

    Args:
        n: Total number of indices
        k_validate: Number of items in the second set
        k_test: Number of items in the third set
        exclude: optional, list of indices to exclude

    Returns:
        train: First set (remaining items)
        validate: Second set
        test: Third set
    """

    full = generate_indices(n, exclude)

    model_building, test = generate_distinct_sets(full, k_test)
    rest, validate = generate_distinct_sets(model_building, k_validate)

    return rest, validate, test



def twoway_split(n, k, exclude=[]):
    """Generate two distinct index sets, one with n-k and one with k items.

    Items are picked at random and not repeated.

    In this split, the remaining items are returned first.

    Args:
        n: Total number of indices
        k: Number of items in the second set
        exclude: optional, list of indices to exclude

    Returns:
        rest: First set (remaining items)
        picked: Second set
    """

    if isinstance(n, int):
        full = np.arange(n)
    else:
        full = n

    full = np.setdiff1d(full, exclude)

    rest, picked = generate_distinct_sets(full, k)

    return rest, picked



def generate_indices(n, exclude):
    """If n is an integer, generate range(0, n), otherwise don't. Then exclude indices from exclude. """

    if isinstance(n, int):
        full = np.arange(n)
    else:
        full = n

    full = np.setdiff1d(full, exclude)

    return full



def generate_distinct_sets(full, k):
    """Out of a full set with n items, pick two sub-sets with n-k and k items at random."""

    picked = np.random.choice(full, size=k, replace=False)
    rest = np.setdiff1d(full, picked, assume_unique=True)

    return rest, picked
