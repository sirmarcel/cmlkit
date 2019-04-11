import os
import numpy as np
import qmmlpack as qmml

from .engine import compute_hash, Configurable, save_npy
from .convert import convert


def from_old_datset(data):
    if data._type == 'Dataset':
        return Dataset(data.z, data.r, b=data.b, p=data.p, name=data.id, desc=data.desc)
    elif data._type == 'Subset':
        return Subset(data.z, data.r, b=data.b, p=data.p, name=data.id, desc=data.desc, idx=data.idx, parent_info=data.parent_info)
    else:
        raise ValueError("No idea what to do with {}!".format(data))


# TODO: Write get_dataset
# def read(filename):
#     d = cmlio.read(filename, ext=False)

#     if 'parent_info' in d.keys():
#         return Subset.from_dict(d)
#     else:
#         return Dataset.from_dict(d)


class Dataset(Configurable):
    """Dataset"""

    kind = 'dataset'

    def __init__(self, z, r, b=None, p={}, name=None, desc=''):
        super().__init__()

        # Sanity checks
        assert len(z) == len(r), \
            'Attempted to create dataset, but z and r are not of the same size ({} vs {})!'.format(len(z), len(r))
        assert b is None or len(b) == len(z), \
            'Attempted to create dataset, but z and b are not of the same size ({} vs {})!'.format(len(z), len(b))
        assert len(r) > 0, 'Attempted to create dataset, r has 0 length!'

        self.desc = desc
        self.z = z
        self.r = r
        self.b = b
        self.p = p

        if name is None:
            name = compute_hash(self.z, self.r, self.b, self.p)
        self.name = name

        self.n = len(self.z)

        self._report = None
        self._info = None
        self._incidence = None  # n_structures x n_atoms_total matrix; 1 if an atom belongs to a given structure

        # compute auxiliary info that we need to convert properties
        self.aux = {}
        n_atoms = np.array([len(zz) for zz in self.z])  # count atoms in unit cell
        n_non_O = np.array([len(zz[zz != 8]) for zz in self.z])  # count atoms that are not Oxygen
        n_non_H = np.array([len(zz[zz != 1]) for zz in self.z])  # count atoms that are not Oxygen

        self.aux['n_atoms'] = n_atoms
        self.aux['n_non_O'] = n_non_O
        self.aux['n_non_H'] = n_non_H

    def __getitem__(self, idx):
        return View(self, idx)

    def __str__(self):
        return "{} named {}; {} (hash {})".format(self.__class__.kind, self.name, self.desc, compute_hash(compute_hash(self.z, self.r, self.b, self.p)))

    def __eq__(self, other):
        hash_equal = (hash(self) == hash(other))
        # TODO: this seems to give false negatives in some cases, but I cannot figure out why.
        # so PROCEED WITH CAUTION
        return hash_equal

    def __hash__(self):
        return int(compute_hash(self.get_config()), 16)  # python expects integer hash digest instead of hex

    def __getstate__(self):
        # this is what gets called when pickling;
        # we delete the lazily-computed attributes,
        # since they would otherwise change the hash
        # computed by joblib, which uses pickle to obtain
        # a serialised representation.
        state = self.__dict__.copy()
        state['_incidence'] = None
        state['_report'] = None
        state['_info'] = None
        return state

    def is_equal(self, other, z=True, r=True, b=True, p=True, print_report=False):
        checks = []
        report = ''

        if z:
            res = self._compare(self.z, other.z)
            report += f"z: {res} "
            checks.append(res)

        if r:
            res = self._compare(self.r, other.r)
            report += f"r: {res} "
            checks.append(res)

        if b:
            res = self._compare(self.b, other.b)
            report += f"b: {res} "
            checks.append(res)

        if p:
            res = self._compare(self.p, other.p)
            report += f"p: {res} "
            checks.append(res)

        if print_report:
            print(report)

        return all(checks)

    @property
    def info(self):
        if self._info is not None:
            return self._info
        else:
            self._info = compute_dataset_info(self)
            return self._info

    @property
    def incidence(self):
        if self._incidence is not None:
            return self._incidence
        else:
            self._incidence = compute_incidence(self)
            return self._incidence

    def _compare(self, a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            # this is monstrous, but it catches the case of ragged
            # object arrays, where array_equal seems to just not work
            if a.shape != b.shape:
                return False

            for i in range(len(a)):
                try:
                    res = np.array_equal(a[i], b[i])
                    if res is False:
                        return False
                except (IndexError, KeyError):
                    return False
            return True

        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            else:
                return all([self._compare(a[k_a], b[k_a]) for k_a in a.keys()])

        else:
            return a == b

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls(config['z'], config['r'], b=config['b'], p=config['p'], name=config['name'], desc=config['desc'])

    def _get_config(self):

        d = {
            'name': self.name,
            'desc': self.desc,
            'z': self.z,
            'r': self.r,
            'b': self.b,
            'p': self.p
        }

        return d

    def save(self, dirname='', filename=None):
        """Save to disk, defaulting to the name as filename"""

        if filename is None:
            filename = self.name

        save_npy(os.path.join(dirname, filename + '.data'), self.get_config())

    def pp(self, target, per='None'):
        return convert(self, self.p[target], per=per)

    @property
    def report(self):

        if self._report is None:
            i = self.info
            general = '# {}: {} #\n\n'.format(self.__class__.kind, self.name) + self.desc + '\n'

            over = '\n## Overview ##\n'
            if self.b is None:
                over += ' {} finite systems (molecules)'.format(i['number_systems']) + '\n'
            else:
                over += ' {} periodic systems (materials)'.format(i['number_systems']) + '\n'
            keys = [str(k) for k in self.p.keys()]
            over += ' {} different properties: {}\n'.format(len(self.p.keys()), keys)

            elems = ' elements: {} ({})'.format(' '.join([qmml.element_data(el, 'abbreviation') for el in i['elements']]), len(i['elements'])) + '\n'
            elems = ' elements by charge: {}'.format(i['elements']) + '\n'
            elems += ' max #els/system: {};  max same #el/system: {};  max #atoms/system: {}'.format(i['max_elements_per_system'], i['max_same_element_per_system'], i['max_atoms_per_system']) + '\n'

            dist = ' min dist: {:3.2f};  max dist: {:3.2f}'.format(i['min_distance'], i['max_distance']) + '\n'

            g = i['geometry']
            geom = '\n## Geometry ##'
            geom += '\n### Ranges ###\n'
            geom += ' These are the ranges for various geometry properties.\n'
            geom += ' count   : {} to {}'.format(g['min_count'], g['max_count']) + '\n'
            geom += ' dist    : {:4.4f} to {:4.4f}'.format(g['min_dist'], g['max_dist']) + '\n'
            geom += ' 1/dist  : {:4.4f} to {:4.4f}'.format(g['min_1/dist'], g['max_1/dist']) + '\n'
            geom += ' 1/dist^2: {:4.4f} to {:4.4f}'.format(g['min_1/dist^2'], g['max_1/dist^2']) + '\n'
            geom += '\n### Recommendations for d ###\n'
            geom += ' We recommend using the intervals (-0.05*max, 1.05*max) for the parametrisation of the MBTR, i.e. a 5% padding. '
            geom += ' In the following, n is the number of bins.\n'
            geom += ' k=1 MBTR:\n'
            geom += ' count     : ({:4.2f}, {:4.2f}/n, n)'.format(-0.05 * g['max_count'], 1.1 * g['max_count']) + '\n'
            geom += ' k=2 MBTR:\n'
            geom += ' 1/dist    : ({:4.2f}, {:4.2f}/n, n)'.format(-0.05 * g['max_1/dist'], 1.1 * g['max_1/dist']) + '\n'
            geom += ' 1/dot     : ({:4.2f}, {:4.2f}/n, n)'.format(-0.05 * g['max_1/dist^2'], 1.1 * g['max_1/dist^2']) + '\n'
            geom += ' k=3 MBTR (experimental):\n'
            geom += ' angle     : ({:4.2f}, {:4.2f}/n, n)'.format(-0.05 * np.pi, 1.1 * np.pi) + '\n'
            geom += ' cos_angle : ({:4.2f}, {:4.2f}/n, n)'.format(-1.05 * 1, 2.1) + '\n'
            geom += ' dot/dotdot: ({:4.2f}, {:4.2f}/n, n)'.format(-0.05 * g['max_1/dist^2'], 1.1 * g['max_1/dist^2']) + '\n'
            geom += ' It is still prudent to experiment with these settings!\n'

            p = i['properties']
            prop = '\n## Properties ##\n'
            prop += ' Mean and standard deviation of properties:\n'
            for k, v in p.items():
                prop += ' {}: {:4.4f} ({:4.4f})\n'.format(k, v[0], v[1])

            self._report = general + over + elems + dist + geom + prop

        return self._report


class Subset(Dataset):
    """Subset of data from a Dataset"""

    kind = 'subset'

    def __init__(self, z, r, b=None, p={}, name=None, desc='', idx=None, parent_info={}):
        # you probably want to use from_dataset in 99% of cases
        super().__init__(z, r, b, p, name=name, desc=desc)

        self.idx = idx
        self.parent_info = parent_info

    @classmethod
    def from_dataset(cls, dataset, idx, name=None, desc=''):

        n = len(idx)

        z = dataset.z[idx]
        r = dataset.r[idx]
        if dataset.b is not None:
            b = dataset.b[idx]
        else:
            b = None

        sub_properties = {}

        for p, v in dataset.p.items():
            sub_properties[p] = v[idx]

        p = sub_properties

        if desc == '':
            desc = "Subset of dataset {} with n={} entries".format(dataset.name, n)

        if name is None:
            name = dataset.name + '_subset' + str(n)

        parent_info = {'desc': dataset.desc, 'name': dataset.name}

        return cls(z, r, b=b, p=p, name=name, desc=desc, idx=idx, parent_info=parent_info)

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls(config['z'], config['r'], b=config['b'], p=config['p'], name=config['name'], desc=config['desc'], idx=config['idx'], parent_info=config['parent_info'])

    def _get_config(self):

        d = {
            'name': self.name,
            'desc': self.desc,
            'z': self.z,
            'r': self.r,
            'b': self.b,
            'p': self.p,
            'idx': self.idx,
            'parent_info': self.parent_info
        }

        return d


class View():
    """View onto a Dataset

    This class is intended to be used when only parts of
    a Dataset need to be accessed, but no permanent Subset
    is required, for instance when chunking a bigger set.
    There should be no need to create instances of this class
    by hand, it gets created by Dataset.__getitem__.

    While Dataset and its subclasses are used for saving and
    loading, this one is the actual workhorse that gets consumed
    by the other machinery.

    """

    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset

        if isinstance(idx, int):
            # we are looking at a single item!
            idx = [idx]
        self.idx = idx

        self.n = len(idx)

        self.desc = "view_{}_n_{}".format(dataset.name, self.n)
        self.desc = "View on dataset {} with n={} entries".format(dataset.name, self.n)

        self._info = None
        self._incidence = None
        self._mask = None  # mask selecting atoms belonging to this view
        # CAVEAT: the mask has no ordering, so you need to
        # ensure that idx is ordered so the structure -> property mapping
        # is maintained!

    @property
    def z(self):
        return self.dataset.z[self.idx]

    @property
    def r(self):
        return self.dataset.r[self.idx]

    @property
    def b(self):
        if self.dataset.b is not None:
            return self.dataset.b[self.idx]
        else:
            return self.dataset.b

    @property
    def p(self):
        return DictView(self.dataset.p, self.idx)

    @property
    def aux(self):
        return DictView(self.dataset.aux, self.idx)

    # TODO: This should not be duplicated here!
    @property
    def info(self):
        if self._info is not None:
            return self._info
        else:
            self._info = compute_dataset_info(self)
            return self._info

    @property
    def incidence(self):
        if self._incidence is not None:
            return self._incidence
        else:
            self._incidence = compute_incidence(self)
            return self._incidence

    @property
    def mask(self):
        if self._mask is not None:
            return self._mask
        else:
            self._mask = compute_mask(self.dataset.incidence, self.idx)
            return self._mask


class DictView(dict):
    """View on a dictionary where each value is an ndarray"""

    def __init__(self, d, idx):
        super(DictView, self).__init__(**d)
        self.d = d
        self.idx = idx

    def __getitem__(self, key):
        return self.d[key][self.idx]


def compute_dataset_info(dataset):
    """Information about a dataset.

    Returns a dictionary containing information about a dataset.

    Args:
      dataset: dataset

    Returns:
      i: Dict with the following keys:
          elements: elements occurring in dataset
          max_elements_per_system: largest number of different elements in a system
          max_same_element_per_system: largest number of same-element atoms in a system
          max_atoms_per_system: largest number of atoms in a system
          min_distance: minimum distance between atoms in a system
          max_distance: maximum distance between atoms in a system
          geometry: additional detailed info about geometries (see below)
    """
    z = dataset.z
    r = dataset.r
    p = dataset.p

    i = {}

    i['number_systems'] = len(z)

    # elements
    i['elements'] = np.unique(np.asarray([a for s in z for a in s], dtype=np.int))  # note that this is always sorted
    i['total_elements'] = len(i['elements'])
    i['max_elements_per_system'] = max([np.nonzero(np.bincount(s))[0].size for s in z])
    i['max_same_element_per_system'] = max([max(np.bincount(s)) for s in z])
    i['min_same_element_per_system'] = min([min(np.bincount(s)) for s in z])

    # systems
    i['max_atoms_per_system'] = max([len(s) for s in z])
    i['systems_per_element'] = np.asarray([np.sum([1 for m in z if el in m]) for el in range(118)], dtype=np.int)

    # atoms
    i['atoms_by_system'] = np.array([len(s) for s in z], dtype=int)
    i['total_atoms'] = np.sum(i['atoms_by_system'])

    # distances
    dists = [qmml.lower_triangular_part(qmml.distance_euclidean(rr), -1) for rr in r]
    i['min_distance'] = min([min(d) for d in dists if len(d) > 0])
    i['max_distance'] = max([max(d) for d in dists if len(d) > 0])

    # geometry info
    geom = {}
    geom['max_dist'] = i['max_distance']
    geom['min_dist'] = i['min_distance']

    geom['max_1/dist'] = 1 / geom['min_dist']
    geom['max_1/dist^2'] = 1 / geom['min_dist']**2

    geom['min_1/dist'] = 1 / geom['max_dist']
    geom['min_1/dist^2'] = 1 / geom['max_dist']**2

    geom['max_count'] = i['max_same_element_per_system']
    geom['min_count'] = i['min_same_element_per_system']

    i['geometry'] = geom

    # property info
    prop = {}
    for k, v in p.items():
        prop[k] = (np.mean(v), np.std(v))

    i['properties'] = prop

    return i


def compute_incidence(dataset):
    """Compute the atomic incidence matrix of a dataset

        This is a n x total_atoms matrix which is one wherever
        an atom belongs to a given structure (needed for predictions
        with atomic contributions instead of whole structures).

        """

    total_atoms = dataset.info['total_atoms']
    incidence = np.zeros((dataset.n, total_atoms), dtype=int)
    pos = 0
    for i, z in enumerate(dataset.z):
        n_atoms = len(z)
        incidence[i, pos:pos + n_atoms] = 1
        pos += n_atoms

    return incidence


def compute_mask(incidence, idx):
    """Generate a mask selecting atoms from structures in idx"""

    mask = np.zeros_like(incidence[0])

    for i in idx:
        mask += incidence[i]

    return np.where(mask == 1)
