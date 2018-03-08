import numpy as np
import yaml


def save(outfile, d):
    """Save the dictionary d to oufile.

    Note that .npy will automatically be appended to the filename.
    """

    np.save(outfile, d)


def read(file, ext=True):
    """Load numpy data stored in file.

    Args:
        file: file to open
        ext: If True, append .npy to filename

    Returns:
        d: dict with data from file
    """
    if ext:
        d = np.load(file + '.npy').item()
    else:
        d = np.load(file).item()

    return d


def read_yaml(filename):
    """Read yaml dictionary from file."""

    stream = open(filename, 'r')
    d = yaml.load(stream)
    stream.close()

    return d
