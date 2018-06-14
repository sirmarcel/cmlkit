from cmlkit import load_dataset
import numpy as np
from cmlkit.model_spec import ModelSpec
from cmlkit.reps.cached_mbtr import DiskAndMemCachedMBTR
from cmlkit.utils.timing import timerfunc

d = load_dataset('kaggle')

mbtr1 = {'acc': 0.001,
         'geomf': 'count',
         'd': [-0.5, 1.6333333333333333, 30],
         'distrf': ('normal', 0.1),
         'aindexf': 'full',
         'eindexf': 'full',
         'elems': None,
         'flatten': True,
         'k': 1,
         'norm': 1.0,
         'weightf': 'identity'}


mbtr2_dist = {'acc': 0.001,
              'geomf': '1/distance',
              'd': (-0.04, 0.008, 100),
              'distrf': ('normal', 0.1),
              'aindexf': 'noreversals',
              'eindexf': 'noreversals',
              'elems': None,
              'flatten': True,
              'k': 2,
              'norm': None,
              'weightf': 'identity^2'}

mbtr2_dist_norm = {'acc': 0.001,
                   'geomf': '1/distance',
                   'distrf': ('normal', 0.1),
                   'd': [-0.04, 0.008, 100],
                   'aindexf': 'noreversals',
                   'eindexf': 'noreversals',
                   'elems': None,
                   'flatten': True,
                   'k': 2,
                   'norm': 1.0,
                   'weightf': 'identity^2'}


spec_dict = {'name': 'k1_t',
             'desc': 'Test caching my dude',
             'data': {'n_train': 1900,
                      'n_valid': 400,
                      'name': 'kaggle',
                      'subset': 'nodup'},
             'krr': {'centering': False,
                     'kernelf': ('gaussian', 100),
                     'nl': 1e-5},
             'mbtrs': {'mbtr_1': mbtr1, 'mbtr_2': mbtr2_dist}
             }

spec_dict2 = {'name': 'k1_t2',
              'desc': 'Test caching my dude',
              'data': {'n_train': 1900,
                       'n_valid': 400,
                       'name': 'kaggle',
                       'subset': 'nodup'},
              'krr': {'centering': False,
                      'kernelf': ('gaussian', 100),
                      'nl': 1e-5},
              'mbtrs': {'mbtr_1': mbtr1, 'mbtr_2': mbtr2_dist_norm}
              }


spec = ModelSpec.from_dict(spec_dict)
spec2 = ModelSpec.from_dict(spec_dict2)


@timerfunc
def f1():
    return DiskAndMemCachedMBTR(d, spec)


@timerfunc
def f2():
    return DiskAndMemCachedMBTR(d, spec2)


m, t1 = f1()  # write to disk
m, t1b = f1()  # read from disk, cache in memory
assert(t1 > t1b)
m, t2 = f2()  # read from disk, compute norm
assert(t2 < t1)
assert(t2 > t1b)
m, t2b = f2()  # read from memory
assert(t2b < t2)