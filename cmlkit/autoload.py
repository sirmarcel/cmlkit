import os
from cmlkit.dataset import read

if 'CML_DATASET_PATH' in os.environ:
    storage_path = [os.path.normpath(p) for p in str(os.environ['CML_DATASET_PATH']).split(':')]
else:
    storage_path = []

storage_path.append(os.path.normpath(os.environ['PWD']))


def load_dataset(name):
    """Load a dataset with given name

    Attempts to automatically load a dataset with the given
    file name. The idea here is that you set a global location
    where all datasets are stored as environment variable CML_DATASET_PATH,
    formatted like the normal PATH variable, i.e. /my/first/path:/my/second/path.

    As last resort, an empty path will be tried, which should default to the local directory.

    Args:
        name: Filename of dataset

    Returns:
        Instance of Dataset or Subset

    """
    # This implicitly loads the first dataset found
    for p in storage_path:
        try:
            file = os.path.join(p, name + '.dat.npy')
            return read(file)
        except FileNotFoundError:
            pass

    raise FileNotFoundError('Could not find dataset {} in paths {}'.format(name, storage_path))
