import os
from qmmltools.dataset import read

if 'QMML_DATASET_PATH' in os.environ:
    storage_path = [os.path.normpath(p) for p in str(os.environ['QMML_DATASET_PATH']).split(':')]
    storage_path.append('')
else:
    storage_path = ['']


def dataset(name, subset=''):
    """Load a dataset with given name

    Attempts to automatically load a dataset with the given
    file name. The idea here is that you set a global location
    where all datasets are stored as environment variable QMML_DATASET_PATH,
    formatted like the normal PATH variable, i.e. /my/first/path:/my/second/path.

    As last resort, no path will be tried, which should default to the local directory.

    """
    for p in storage_path:
        if subset is not '':
            file = os.path.join(p, name + '-' + subset + '.dat.npy')
        else:
            file = os.path.join(p, name + '.dat.npy')

        return read(file)
