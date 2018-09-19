import os
import cmlkit.dataset as cmld
from cmlkit.utils.caching import _memcached

if 'CML_DATASET_PATH' in os.environ:
    storage_path = [os.path.normpath(p) for p in str(os.environ['CML_DATASET_PATH']).split(':')]
else:
    storage_path = []

storage_path.append(os.path.normpath(os.environ['PWD']))

cached_reader = _memcached(cmld.read, max_entries=10)


def load_dataset(name, reader=cmld.read):
    """Load a dataset with given (file) name

    Attempts to automatically load a dataset with the given
    (file) name. If a path to a dataset is passed, this is loaded,
    if not, we will look for a dataset with that name in the PATHs
    specified in $CML_DATASET_PATH. This path always also includes
    the local working directory, so omitting the '.dat.npy' extension
    will also work.

    Args:
        name: Name of dataset, or path to dataset

    Returns:
        Instance of Dataset or Subset

    """
    # First, try if you have passed a fully formed dataset path
    if os.path.isfile(name):
        return reader(name)

    # If we have a dataset name (and not a file name), add the default extension
    if '.dat.npy' not in name:
        name += '.dat.npy'

    # Go through the dataset paths, return the first dataset found
    for p in storage_path:
        try:
            file = os.path.join(p, name)
            return reader(file)
        except FileNotFoundError:
            pass

    raise FileNotFoundError('Could not find dataset {} in paths {}'.format(name, storage_path))


def load_dataset_cached(name):
    return load_dataset(name, reader=cached_reader)
