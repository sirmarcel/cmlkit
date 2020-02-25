from pathlib import Path

from cmlkit.dataset import Dataset, Subset
from cmlkit.engine import _from_npy
from cmlkit.env import dataset_path

classes = {Subset.kind: Subset, Dataset.kind: Dataset}


def load_dataset(name, other_paths=[]):
    """Load a dataset with given (file) name."""
    if isinstance(name, Dataset):
        return name

    path = Path(name)

    # First, try if you have passed a fully formed dataset path
    if path.is_file():
        return _from_npy(name, classes=classes)

    # Go through the dataset paths, return the first dataset found
    all_paths = dataset_path + other_paths
    for p in all_paths:
        try:
            file = p / path
            return _from_npy(file, classes=classes)
        except FileNotFoundError:
            pass

    raise FileNotFoundError(
        "Could not find dataset {} in paths {}".format(name, all_paths)
    )
