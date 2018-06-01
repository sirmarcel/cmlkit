from cmlkit.dataset import read
from cmlkit.mbtr.mbtr import MBTR
from cmlkit.model_spec import ModelSpec

m = ModelSpec.from_yaml('model_mini.spec.yml')
d = read('test.dat.npy')

mbtr = MBTR(d, m)
mbtr.save()
