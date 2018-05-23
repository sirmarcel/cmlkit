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
        filename = file + '.npy'

    else:
        filename = file

    d = np.load(filename)

    if d.size == 1:
        return d.item()
    else:
        return d


def read_yaml(filename):
    """Read yaml dictionary from file."""

    stream = open(filename, 'r')
    d = yaml.load(stream)
    stream.close()

    return d


def save_yaml(filename, d):
    """Save a dict as yaml.

    Note that .yml will automatically be appended to the filename.

    Formatting is done as follows: Dicts are NOT expressed in flowstyle,
    i.e. newlines for dictionary keys, but tuples and lists are done in
    flowstyle, i.e. inline.

    Args:
        filename: Path to file
        d: Dict to save
        flowstyle: Bool indicating whether dicts are saved compactly or not

    Returns: None

    """

    # Define representer for lists/tuples
    def sequence_representer(dumper, data):
        return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

    # Register the above representer for tuples and lists
    yaml.add_representer(tuple, sequence_representer)
    yaml.add_representer(list, sequence_representer)

    with open(filename + '.yml', 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)
