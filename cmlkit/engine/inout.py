import numpy as np
import yaml
import son
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


# register some custom dumpers for yaml
# weird that this can only be done at module level!

# Define representer for lists/tuples
def sequence_representer(dumper, data):
    return dumper.represent_sequence(u"tag:yaml.org,2002:seq", data, flow_style=True)


# Define representer for numpy floats
def float_representer(dumper, data):
    return dumper.represent_float(data)


# Define representer for numpy ints
def int_representer(dumper, data):
    return dumper.represent_int(data)


# Register the above representers
yaml.add_representer(tuple, sequence_representer)
yaml.add_representer(list, sequence_representer)
yaml.add_representer(np.float64, float_representer)
yaml.add_representer(np.int64, int_representer)


def normalize_extension(path, extension):
    """If the path doesn't have the extension, add it."""
    p = Path(path)
    return p.with_suffix(extension)


def makedir(p):
    """Create directory at path and its parents.

    Args:
        p: Path-like object
    """
    path = Path(p)
    path.mkdir(exist_ok=True, parents=True)


def save_npy(filename, d):
    """Save a dictionary with numpy.

    Args:
        filename: Path-like object. (Extension not required.)
        d: Dict to save.
    """
    np.save(normalize_extension(filename, ".npy"), d)


def safe_save_npy(filename, d):
    """Save a dict with numpy using a separate Thread.

    This is important because otherwise corrupted files are written to disk,
    when the pickler gets interrupted mid-write.

    Args:
        filename: Path-like object. (Extension not required.)
        d: Dict to save.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(save_npy, filename, d)


def read_npy(filename):
    """Load numpy data stored in filename."""

    d = np.load(normalize_extension(filename, ".npy"), allow_pickle=True)

    if d.size == 1:
        return d.item()
    else:
        return d


def save_yaml(filename, d):
    """Save a dict as yaml.

    Formatting is done as follows: Dicts are NOT expressed in flowstyle,
    i.e. newlines for dictionary keys, but tuples and lists are done in
    flowstyle, i.e. inline.

    Args:
        filename: Path to file. (Extension not required.)
        d: Dict to save.

    """

    with open(normalize_extension(filename, ".yml"), "w") as outfile:
        yaml.dump(d, outfile, default_flow_style=False)


def read_yaml(filename):
    """Read yaml dictionary from filename."""

    with open(normalize_extension(filename, ".yml"), "r") as stream:
        d = yaml.safe_load(stream)

    return d


def save_son(filename, d, is_metadata=False):
    """Save object into a SON file."""

    son.dump(
        d, normalize_extension(filename, ".son"), is_metadata=is_metadata, dumper=yaml.dump
    )


def read_son(filename):
    """Load from a SON file."""

    return son.load(normalize_extension(filename, ".son"), loader=yaml.safe_load)
