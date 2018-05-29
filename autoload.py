import os
from qmmltools.dataset import read

if 'QMML_DATASET_PATH' in os.environ:
    storage_path = [os.path.normpath(p) for p in str(os.environ['QMML_DATASET_PATH']).split(':')]
    storage_path.append('')
else:
    storage_path = ['']


def load_dataset(name):
    """Load a dataset with given name

    Attempts to automatically load a dataset with the given
    file name. The idea here is that you set a global location
    where all datasets are stored as environment variable QMML_DATASET_PATH,
    formatted like the normal PATH variable, i.e. /my/first/path:/my/second/path.

    As last resort, an empty path will be tried, which should default to the local directory.

    Args:
        name: Filename of dataset

    Returns:
        Instance of Dataset or Subset

    """
    # This implicitly loads the first dataset found
    for p in storage_path:
        file = os.path.join(p, name + '.dat.npy')
        return read(file)
