from qmmltools.dataset import read
from qmmltools.mbtr.mbtr import MBTR
from qmmltools.model_spec import ModelSpec

m = ModelSpec.from_yaml('model_mini.spec.yml')
d = read('test.dat.npy')

mbtr = MBTR(d, m)
mbtr.save()
