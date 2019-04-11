import numpy as np
from cmlkit import cache_location
from ..engine import BaseComponent, memcached, diskcached


def compute_distances(r):
    n_mols = r.shape[0]
    n_atoms = r.shape[1]  # number of atoms per structure
    n_dists = int((n_atoms - 1) * n_atoms / 2)  # Number of unique combinations of atoms
    result = np.zeros((n_mols, n_dists))
    for i, rs in enumerate(r):
        result[i] = _single_mol(rs)

    return result


def _single_mol(rs):
    n = len(rs)
    # result = np.zeros(n*(n+1)/2)

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(np.linalg.norm(rs[i] - rs[j]))

    return np.array(result)


def compute_histograms(all_dists, d, broadening):

    grid = np.linspace(d[0], d[1], num=d[2])
    result = np.zeros((all_dists.shape[0], d[2]))

    for i, dists in enumerate(all_dists):
        for dist in dists:
            result[i] += np.exp(-(dist - grid)**2 / broadening)

    return result


class OnlyDists(BaseComponent):
    """A representation which is simply all distances 

    Like OnlyCoords this will not work for multiple molecules/structures,
    it is purely for conformations."""

    kind = 'only_dists'

    def __init__(self, context={}):
        super().__init__(context=context)
        self.cache_type = self.context['cache_type']
        self.min_duration = self.context['min_duration']

        self.computer = compute_distances
        cache_entries = 25

        if self.cache_type == 'mem':
            self.computer = memcached(self.computer, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            disk_cached = diskcached(self.computer, cache_location=cache_location, name='dists', min_duration=self.min_duration)
            self.computer = memcached(disk_cached, max_entries=cache_entries)

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(context=context)

    def _get_config(self):
        return {}

    def compute(self, data):

        return self.computer(data.r)


class OnlyDistsHistogram(BaseComponent):
    """A representation which is simply all distances, smeared out as histogram

    This is basically a poor man's MBTR2, but will only work under the
    same constraints as OnlyDists since we piggyback on the distance computer there."""

    kind = 'only_dists_histogram'

    def __init__(self, d=[0.0, 10.0, 100], broadening=1.0, context={}):
        super().__init__(context=context)
        self.cache_type = self.context['cache_type']
        self.min_duration = self.context['min_duration']

        self.d = d  # discretisation settings: min, max, n
        self.broadening = broadening  # magnitude of gaussian smearing

        self.dist_computer = compute_distances
        self.hist_computer = compute_histograms
        cache_entries = 25

        if self.cache_type == 'mem':
            self.dist_computer = memcached(self.dist_computer, max_entries=cache_entries)
            self.hist_computer = memcached(self.hist_computer, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            dist_disk_cached = diskcached(self.dist_computer, cache_location=cache_location, name='dists', min_duration=self.min_duration)
            self.dist_computer = memcached(dist_disk_cached, max_entries=cache_entries)

            hist_disk_cached = diskcached(self.hist_computer, cache_location=cache_location, name='hists', min_duration=self.min_duration)
            self.hist_computer = memcached(hist_disk_cached, max_entries=cache_entries)

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {
            'd': self.d,
            'broadening': self.broadening
        }

    def compute(self, data):
        all_dists = self.dist_computer(data.r)

        return self.hist_computer(all_dists, self.d, self.broadening)
