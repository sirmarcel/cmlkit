from pathlib import Path, PurePath

from .dataset import Dataset, Subset
from .engine import _from_npy
from .env import dataset_path

classes = {
    Subset.kind: Subset,
    Dataset.kind: Dataset,
}


def load_dataset(name, other_paths=[]):
    """Load a dataset with given (file) name"""
    if isinstance(name, Dataset):
        return name

    path = Path(name)

    # First, try if you have passed a fully formed dataset path
    if path.is_file():
        return _from_npy(name, classes=classes)

    # If we have a dataset name (and not a file name), add the default extension
    if '.data' not in path.suffixes and '.npy' not in path.suffixes:
        path = path.with_suffix('.data.npy')

    # Go through the dataset paths, return the first dataset found
    all_paths = dataset_path + other_paths
    for p in all_paths:
        try:
            file = Path(p) / path
            return _from_npy(file, classes=classes)
        except FileNotFoundError:
            pass

    raise FileNotFoundError('Could not find dataset {} in paths {}'.format(name, dataset_path))
