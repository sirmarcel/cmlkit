from qmmltools.dataset import Dataset, Subset
import numpy as np

z = np.array([[1, 2, 3], [1, 2, 3]])
r = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.1, 0.0], [1.0, 0.0, 0.0], [0.0, 1.1, 0.0]]])
b = None
name = 'test'
desc = 'just a test'
p = {'e': np.array([1.0, 1.1])}

d = Dataset(name, z, r, b, p, desc=desc)
d.save()

sub = Subset(d, [1], 'sub', 'test subset')
sub.save()

from datasets.autoload import load
k = load('kaggle')

idx = np.arange(20)

sub = Subset(k, idx, 'mini', 'small subset of kaggle')
sub.save()
