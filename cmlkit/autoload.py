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
    """Load a dataset with given name

    Attempts to automatically load a dataset with the given
    file name. The idea here is that you set a global location
    where all datasets are stored as environment variable CML_DATASET_PATH,
    formatted like the normal PATH variable, i.e. /my/first/path:/my/second/path.

    As last resort, the current working directory will be tried.

    Args:
        name: Filename of dataset

    Returns:
        Instance of Dataset or Subset

    """
    # This implicitly loads the first dataset found
    for p in storage_path:
        try:
            file = os.path.join(p, name + '.dat.npy')
            return reader(file)
        except FileNotFoundError:
            pass

    raise FileNotFoundError('Could not find dataset {} in paths {}'.format(name, storage_path))


def load_dataset_cached(name):
    return load_dataset(name, reader=cached_reader)
