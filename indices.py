import numpy as np


def fourway_split(n, k_train, k_validate, k_test):
    """Split the total n indices into four subsets.

    Items are picked at random and not repeated.

    Args:
        n: Total number of indices
        k_train: Number of items in the first set
        k_validate: Number of items in the second set
        k_test: Number of items in the third set

    Returns:
        train: First set
        validate: Second set
        test: Third set
        rest: Fourth set with remaining items
    """

    full = np.arange(n)
    rest, train = generate_distinct_sets(full, k_train)
    rest, validate = generate_distinct_sets(rest, k_validate)
    rest, test = generate_distinct_sets(rest, k_test)

    return train, validate, test, rest


def threeway_split(n, k_validate, k_test):
    """Generate training, validation and training index sets for n items.

    Items are picked at random and not repeated.

    In this split, the remaining items are returned first.

    Args:
        n: Total number of indices
        k_validate: Number of items in the second set
        k_test: Number of items in the third set

    Returns:
        train: First set (remaining items)
        validate: Second set
        test: Third set
    """

    full = np.arange(n)
    model_building, test = generate_distinct_sets(full, k_test)
    train, validate = generate_distinct_sets(model_building, k_validate)

    return train, validate, test



def generate_distinct_sets(full, k):
    """Out of a full set with n items, pick two sub-sets with n-k and k items at random."""

    picked = np.random.choice(full, size=k, replace=False)
    rest = np.setdiff1d(full, picked, assume_unique=True)

    return rest, picked
