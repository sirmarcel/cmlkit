from cmlkit import load_dataset
import numpy as np
from cmlkit.model_spec import ModelSpec
from cmlkit.reps.cached_mbtr import DiskAndMemCachedMBTR

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

mbtr1b = {'acc': 0.001,
          'geomf': 'count',
          'd': [-0.5, 1.6333333333333333, 30],
          'distrf': ('normal', 0.15),
          'aindexf': 'full',
          'eindexf': 'full',
          'elems': None,
          'flatten': True,
          'k': 1,
          'norm': 1.0,
          'weightf': 'identity'}

mbtr1c = {'acc': 0.001,
          'geomf': 'count',
          'd': [-0.5, 1.6333333333333333, 30],
          'distrf': ('normal', 0.15),
          'aindexf': 'full',
          'eindexf': 'full',
          'elems': None,
          'flatten': True,
          'k': 1,
          'norm': None,
          'weightf': 'identity'}


spec_dict = {'name': 'k1_t',
             'desc': 'Test caching my dude',
             'data': {'n_train': 1900,
                      'n_valid': 400,
                      'name': 'kaggle',
                      'subset': 'nodup'},
             'krr': {'centering': False,
                     'kernelf': ('gaussian', 100),
                     'nl': 1e-5},
             'mbtrs': {'mbtr_1': mbtr1, 'mbtr_2': mbtr1b}
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
              'mbtrs': {'mbtr_1': mbtr1b, 'mbtr_2': mbtr1c}
              }


spec = ModelSpec.from_dict(spec_dict)
spec2 = ModelSpec.from_dict(spec_dict2)

# This should generate only 2 files on disk, since norm is only cached in memory
m = DiskAndMemCachedMBTR(d, spec)
m = DiskAndMemCachedMBTR(d, spec2)
m = DiskAndMemCachedMBTR(d, spec)
m = DiskAndMemCachedMBTR(d, spec)
m = DiskAndMemCachedMBTR(d, spec2)
# Re-run this multiple times to check stability against restarts
