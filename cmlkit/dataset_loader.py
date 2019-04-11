from pathlib import Path, PurePath
from cmlkit import dataset_path, from_npy, Dataset


def load_dataset(name, other_paths=[]):
    """Load a dataset with given (file) name"""
    if isinstance(name, Dataset):
        return name

    path = Path(name)

    # First, try if you have passed a fully formed dataset path
    if path.is_file():
        return from_npy(name)

    # If we have a dataset name (and not a file name), add the default extension
    if '.data' not in path.suffixes and '.npy' not in path.suffixes:
        path = path.with_suffix('.data.npy')

    # Go through the dataset paths, return the first dataset found
    all_paths = dataset_path + other_paths
    for p in all_paths:
        try:
            file = Path(p) / path
            return from_npy(file)
        except FileNotFoundError:
            pass

    raise FileNotFoundError('Could not find dataset {} in paths {}'.format(name, dataset_path))
