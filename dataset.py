import os
import numpy as np
import qmmlpack as qmml
import cmlkit.inout as cmlio
from cmlkit.utils.hashing import hash_sortable_dict, hash_arrays


def read(filename):
    d = cmlio.read(filename, ext=False)

    if 'parent_info' in d.keys():
        return Subset.from_dict(d)
    else:
        return Dataset.from_dict(d)


class Dataset(object):
    """Dataset

    Represents a collection of structures/molecules with different properties,
    which can be saved to a file and loaded from it.

    Attributes:
        id: Canonical name; used internally, equal to name for this class
        name: Short, unique name
        desc: Short description
        family: Indicates broad group of data this belongs to (for instance for property conversion)
        z, r, b: As required by qmmlpack (ragged arrays)
        p: Dict of properties, keys are strings, values are ndarrays
        report: String giving lots of information about this dataset
        info: Dict of properties of this dataset
        n: Number of systems in dataset
        hashes: Dict with hashes, will be used for sanity checking

    """

    def __init__(self, name, z, r, b=None, p={}, info=None, desc='', family=None):
        super(Dataset, self).__init__()

        # Sanity checks
        assert len(z) == len(r), \
            'Attempted to create dataset, but z and r are not of the same size ({} vs {})!'.format(len(z), len(r))
        assert b is None or len(b) == len(z), \
            'Attempted to create dataset, but z and b are not of the same size ({} vs {})!'.format(len(z), len(b))
        assert len(r) > 0, 'Attempted to create dataset, r has 0 length!'

        self.name = name
        self.desc = desc
        self.z = z
        self.r = r
        self.b = b
        self.p = p
        self.id = name

        self.hashes = {'p': hash_sortable_dict(self.p),              # hash of properties
                       'geom': hash_arrays(self.z, self.r, self.b)}  # hash of structure description

        self._type = 'Dataset'
        self._general = ''  # Information about this object, overwritten by Subset
        self._report = None

        if info is None:
            self.info = compute_dataset_info(z, r, p)
        else:
            self.info = info

        if family is None:
            self.family = name
        else:
            self.family = family

        self.n = self.info['number_systems']

    def __getitem__(self, idx):
        return View(self, idx)

    def __str__(self):
        return self.report

    def to_dict(self):
        """Return a dictionary representation"""

        d = {
            'name': self.name,
            'desc': self.desc,
            'id': self.id,
            'family': self.family,
            'z': self.z,
            'r': self.r,
            'b': self.b,
            'p': self.p,
            'info': self.info,
            'n': self.n
        }

        return d

    def save(self, dirname='', filename=None):
        """Save to disk"""

        if filename is None:
            filename = self.id

        cmlio.save(os.path.join(dirname, filename + '.dat'), self.to_dict())

    @property
    def report(self):

        if self._report is None:
            i = self.info
            general = '# {}: {} #\n\n'.format(self._type, self.id) + self.desc + self._general + '\n'

            over = '\n## Overview ##\n'
            if self.b is None:
                over += ' {} finite systems (molecules)'.format(i['number_systems']) + '\n'
            else:
                over += ' {} periodic systems (materials)'.format(i['number_systems']) + '\n'
            keys = [str(k) for k in self.p.keys()]
            over += ' {} different properties: {}\n'.format(len(self.p.keys()), keys)

            elems = ' elements: {} ({})'.format(' '.join([qmml.element_data(el, 'abbreviation') for el in i['elements']]), len(i['elements'])) + '\n'
            elems += ' max #els/system: {};  max same #el/system: {};  max #atoms/system: {}'.format(i['max_elements_per_system'], i['max_same_element_per_system'], i['max_atoms_per_system']) + '\n'

            dist = ' min dist: {:3.2f};  max dist: {:3.2f}'.format(i['min_distance'], i['max_distance']) + '\n'

            g = i['geometry']
            geom = '\n## Geometry ##'
            geom += '\n### Ranges ###\n'
            geom += ' These are the ranges for various geometry functions.\n'
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
            geom += ' 1/dist^2  : ({:4.2f}, {:4.2f}/n, n)'.format(-0.05 * g['max_1/dist^2'], 1.1 * g['max_1/dist^2']) + '\n'
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

    @classmethod
    def from_dict(cls, d):
        return cls(d['name'], d['z'], d['r'], d['b'], d['p'], d['info'], d['desc'], d['family'])


class Subset(Dataset):
    """Subset of data from a Dataset

    This class is intended to allow handling subsets of data of a large
    dataset (which might be inconvenient to handle) while still retaining
    some connection to the full dataset.

    Attributes:
        id: Canonical name of this subset, dataset_name-name
        name: Name of this subset
        family: Broad family of this subset
        desc: Description of this subset
        z, r, b, p, report, info: As in Dataset
        parent_info: Dict of info on parent dataset, namely name and info dict

    """

    def __init__(self, dataset, idx, name=None, desc='', restore_data=None):

        if restore_data is None:
            self.idx = idx
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

            self.parent_info = {'info': dataset.info, 'desc': dataset.desc, 'name': dataset.name, 'id': dataset.id}

            if name is not None:
                self.name = name
            else:
                self.name = 'n' + str(n)

            super(Subset, self).__init__(self.name, z, r, b, sub_properties, desc=desc, family=dataset.family)

            self.id = self.parent_info['name'] + '-' + self.name
            self.n = n

        else:
            super(Subset, self).__init__(restore_data['name'],
                                         restore_data['z'],
                                         restore_data['r'],
                                         restore_data['b'],
                                         restore_data['p'],
                                         restore_data['info'],
                                         restore_data['desc'],
                                         restore_data['family'])

            self.parent_info = restore_data['parent_info']
            self.n = restore_data['n']
            self.id = restore_data['id']
            self.idx = restore_data['idx']

        self._type = 'Subset'
        self._general = "\nThis is a subset of the dataset '{}'.".format(self.parent_info['id'])

    def to_dict(self):
        """Return a dictionary representation"""

        d = {
            'name': self.name,
            'desc': self.desc,
            'id': self.id,
            'z': self.z,
            'r': self.r,
            'b': self.b,
            'p': self.p,
            'info': self.info,
            'parent_info': self.parent_info,
            'n': self.n,
            'idx': self.idx,
            'family': self.family
        }

        return d

    @classmethod
    def from_dict(cls, d):
        return cls(None, None, restore_data=d)


class View(object):
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
        super(View, self).__init__()
        self.dataset = dataset
        self.idx = idx
        if isinstance(idx, int):
            self.n = 1
        else:
            self.n = len(idx)

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
    def info(self):
        return self.dataset.info

    @property
    def name(self):
        return self.dataset.name

    @property
    def id(self):
        return self.dataset.id

    @property
    def family(self):
        return self.dataset.family


class DictView(dict):
    """View on a dictionary where each value is an ndarray"""

    def __init__(self, d, idx):
        super(DictView, self).__init__(**d)
        self.d = d
        self.idx = idx

    def __getitem__(self, key):
        return self.d[key][self.idx]


def compute_dataset_info(z, r, p):
    """Information about a dataset.

    Returns a dictionary containing information about a dataset.

    Args:
      z: atomic numbers
      r: atom coordinates, in Angstrom
      p: dict with properties in the dataset (things to predict)

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

    i = {}

    i['number_systems'] = len(z)

    # elements
    i['elements'] = np.unique(np.asarray([a for s in z for a in s], dtype=np.int))
    i['max_elements_per_system'] = max([np.nonzero(np.bincount(s))[0].size for s in z])
    i['max_same_element_per_system'] = max([max(np.bincount(s)) for s in z])
    i['min_same_element_per_system'] = min([min(np.bincount(s)) for s in z])

    # systems
    i['max_atoms_per_system'] = max([len(s) for s in z])
    i['systems_per_element'] = np.asarray([np.sum([1 for m in z if el in m]) for el in range(118)], dtype=np.int)

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
