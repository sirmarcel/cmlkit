import numpy as np
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def makedir(p):
    path = Path(p)
    path.mkdir(exist_ok=True, parents=True)


def save_npy(filename, d):
    """Save a dictionary with numpy"""
    filename = str(filename)
    filename.replace('.npy', '')
    np.save(filename, d)


def safe_save_npy(filename, d):
    """Save a dict with numpy using a separate Thread

    This is important because otherwise corrupted files are written to disk,
    when the pickler gets interrupted mid-write.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(save_npy, filename, d)


def read_npy(filename):
    """Load numpy data stored in filename"""

    d = np.load(filename)

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
        filename: Path to file
        d: Dict to save
        flowstyle: Bool indicating whether dicts are saved compactly or not

    Returns: None

    """
    filename = str(filename)
    filename.replace('.yml', '')

    # Define representer for lists/tuples
    def sequence_representer(dumper, data):
        return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

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

    with open(filename, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)
