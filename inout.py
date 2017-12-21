import numpy as np


def save(outfile, d):
    """Save the dictionary d to oufile.

    Note that .npy will automatically be appended to the filename.
    """

    np.save(outfile, d)


def read(file):
    """Load numpy data stored in file.

    Note that .npy will automatically be appended to the filename.
    """

    d = np.load(file + '.npy').item()
    return d


def read_yaml(filename):
    """Read yaml dictionary from file."""

    stream = open(filename, 'r')
    d = yaml.load(stream)
    stream.close()

    return d
