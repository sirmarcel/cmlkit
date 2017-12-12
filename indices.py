import numpy as np


def threeway_split(n, k_validate, k_test):
    """Generate training, validation and training index sets for n items."""

    full = np.arange(n)
    model_building, test = generate_distinct_sets(full, k_test)
    train, validate = generate_distinct_sets(model_building, k_validate)

    return train, validate, test



def generate_distinct_sets(full, k):
    """Out of a full set with n items, pick two sub-sets with n-k and k items at random."""

    picked = np.random.choice(full, size=k, replace=False)
    rest = np.setdiff1d(full, picked, assume_unique=True)

    return rest, picked
