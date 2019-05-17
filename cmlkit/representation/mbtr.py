from copy import deepcopy
import numpy as np
import qmmlpack as qmml

from ..helpers import convert_sequence
from cmlkit import cache_location
from ..engine import *

# Pure functions for different k-body MBTRs; required to be pure for caching & wrapping


def explicit_single_mbtr_1(z, r, b, d, geomf, weightf, distrf, corrf, eindexf, aindexf, flatten, elems, acc):
    kgwdcea = [1, geomf, weightf, distrf, corrf, eindexf, aindexf]
    return qmml.many_body_tensor(z, r, d, kgwdcea, flatten=flatten, elems=elems, basis=b, acc=acc)


def explicit_single_mbtr_2(z, r, b, d, geomf, weightf, distrf, corrf, eindexf, aindexf, flatten, elems, acc):
    kgwdcea = [2, geomf, weightf, distrf, corrf, eindexf, aindexf]
    return qmml.many_body_tensor(z, r, d, kgwdcea, flatten=flatten, elems=elems, basis=b, acc=acc)


def explicit_single_mbtr_3(z, r, b, d, geomf, weightf, distrf, corrf, eindexf, aindexf, flatten, elems, acc):
    kgwdcea = [3, geomf, weightf, distrf, corrf, eindexf, aindexf]
    return qmml.many_body_tensor(z, r, d, kgwdcea, flatten=flatten, elems=elems, basis=b, acc=acc)


def explicit_single_mbtr_4(z, r, b, d, geomf, weightf, distrf, corrf, eindexf, aindexf, flatten, elems, acc):
    kgwdcea = [4, geomf, weightf, distrf, corrf, eindexf, aindexf]
    return qmml.many_body_tensor(z, r, d, kgwdcea, flatten=flatten, elems=elems, basis=b, acc=acc)


def single_mbtr(data, config, mbtr_gen):
    # This function translates into qmmlpack argument format,
    # and reads out the relevant bits of the dataset, and then
    # dispatches the actual computation to a pure, lower-level function,
    # which can be cached.
    geomf = convert_sequence(config['geomf'])
    weightf = convert_sequence(config['weightf'])
    distrf = convert_sequence(config['distrf'])
    corrf = convert_sequence(config['corrf'])
    eindexf = convert_sequence(config['eindexf'])
    aindexf = convert_sequence(config['aindexf'])
    d = tuple(config['d'])

    norm = config['norm']
    if norm is not None:
        norm = convert_sequence(norm)

    elems = config['elems']
    if elems is not None:
        elems = tuple(elems)

    mbtr = mbtr_gen(data.z, data.r, data.b,
                    d,
                    geomf, weightf, distrf, corrf, eindexf, aindexf,
                    config['flatten'], elems, config['acc'])

    if norm is not None:
        if norm[0] == 'simple':
            for i in range(len(data.z)):
                mbtr[i] /= np.sum(mbtr[i])
            scaled = mbtr * norm[1][0]
            scaled = np.nan_to_num(scaled, copy=False)
            return scaled
    else:
        return mbtr


class BaseMBTR(Component):
    """MBTR Base Class"""
    default_context = {'timeout': None, 'compute_per_structure': False, 'disable_external_wrap': False}

    # mbtr defaults; set per k-body term
    defaults = {}

    def __init__(self, config, context={}):
        super().__init__(context=context)

        self.config = {**self.__class__.defaults, **config}
        self.cache_type = self.context['cache_type']
        self.compute_per_structure = self.context['compute_per_structure']
        self.min_duration = self.context['min_duration']

        # Disables the wrapper for "external calls", which implements the timeout.
        # This is needed because in Python < 3.8, multiprocessing pipes have an intrinsic
        # size limit (see https://bugs.python.org/issue17560), which we sometimes run
        # into. This is of course unfortunate, and I think in the future, timeouts should
        # be handled at a different level of abstraction.
        self.disable_external_wrap = self.context['disable_external_wrap']

        if self.compute_per_structure:
            self.compute = self._compute_per_structure

        self.cache_location = cache_location
        self.timeout = self.context['timeout']  # mbtr computation will be terminated after this number of seconds

        if self.config['norm'] is not None:

            self.norm_type = self.config['norm'][0]

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(config, context=context)

    def _get_config(self):
        return deepcopy(self.config)

    def compute(self, data):
        res = single_mbtr(data, self.config, mbtr_gen=self.mbtr_gen)

        return res

    def _compute_per_structure(self, data):
        # same as compute, but does it per structure (for caching)
        res = []
        for i in range(data.n):
            d = data[i]
            res.append(single_mbtr(d, self.config, mbtr_gen=self.mbtr_gen))

        return np.concatenate(res, axis=0)


class MBTR1(BaseMBTR):
    kind = "mbtr1"
    """One-body MBTR"""
    defaults = {'aindexf': 'full', 'eindexf': 'full', 'corrf': 'identity', 'acc': 0.001, 'flatten': True, 'elems': None, 'norm': None}

    def __init__(self, config, context={}):
        super().__init__(config, context=context)

        self.mbtr_gen = explicit_single_mbtr_1
        if not self.disable_external_wrap:
            self.mbtr_gen = wrap_external(self.mbtr_gen, timeout=self.timeout)

        if self.compute_per_structure:
            cache_entries = 10000
        else:
            cache_entries = 100

        if self.cache_type == 'mem':
            self.mbtr_gen = memcached(self.mbtr_gen, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            disk_cached = diskcached(self.mbtr_gen, cache_location=self.cache_location, name='mbtr_1', min_duration=self.min_duration)
            self.mbtr_gen = memcached(disk_cached, max_entries=cache_entries)

        elif self.cache_type == 'disk':
            disk_cached = diskcached(self.mbtr_gen, cache_location=self.cache_location, name='mbtr_1', min_duration=self.min_duration)


class MBTR2(BaseMBTR):
    kind = "mbtr2"
    """Two-body MBTR"""
    defaults = {'aindexf': 'noreversals', 'eindexf': 'noreversals', 'corrf': 'identity', 'acc': 0.001, 'flatten': True, 'elems': None, 'norm': None}

    def __init__(self, config, context={}):
        super().__init__(config, context=context)

        self.mbtr_gen = explicit_single_mbtr_2
        if not self.disable_external_wrap:
            self.mbtr_gen = wrap_external(self.mbtr_gen, timeout=self.timeout)

        if self.compute_per_structure:
            cache_entries = 10000
        else:
            cache_entries = 50

        if self.cache_type == 'mem':
            self.mbtr_gen = memcached(self.mbtr_gen, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            disk_cached = diskcached(self.mbtr_gen, cache_location=self.cache_location, name='mbtr_2', min_duration=self.min_duration)
            self.mbtr_gen = memcached(disk_cached, max_entries=cache_entries)

        elif self.cache_type == 'disk':
            self.mbtr_gen = diskcached(self.mbtr_gen, cache_location=self.cache_location, name='mbtr_2', min_duration=self.min_duration)


class MBTR3(BaseMBTR):
    kind = "mbtr3"
    """Three-body MBTR"""
    defaults = {'aindexf': 'noreversals', 'eindexf': 'noreversals', 'corrf': 'identity', 'acc': 0.01, 'flatten': True, 'elems': None, 'norm': None}
    default_context = {'timeout': None, 'compute_per_structure': False, 'disable_external_wrap': False}
    # TODO: Fix unintended behaviour of 'timeout' being applied to per-structure
    # calculations individually. In reality, we want timeout = timeout/n_structures
    # when computing, but this needs the ability to pass through a new timeout throught
    # the cache wrappers. Later! For now, this just needs to be done by hand.

    def __init__(self, config, context={}):
        super().__init__(config, context=context)

        self.mbtr_gen = explicit_single_mbtr_3
        if not self.disable_external_wrap:
            self.mbtr_gen = wrap_external(self.mbtr_gen, timeout=self.timeout)

        if self.compute_per_structure:
            cache_entries = 10000
        else:
            cache_entries = 25

        if self.cache_type == 'mem':
            self.mbtr_gen = memcached(self.mbtr_gen, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            disk_cached = diskcached(self.mbtr_gen, cache_location=self.cache_location, name='mbtr_3', min_duration=self.min_duration)
            self.mbtr_gen = memcached(disk_cached, max_entries=cache_entries)

        elif self.cache_type == 'disk':
            self.mbtr_gen = diskcached(self.mbtr_gen, cache_location=self.cache_location, name='mbtr_3', min_duration=self.min_duration)


class MBTR4(BaseMBTR):
    kind = "mbtr4"
    """Four-body MBTR"""
    defaults = {'aindexf': 'noreversals', 'eindexf': 'noreversals', 'corrf': 'identity', 'acc': 0.01, 'flatten': True, 'elems': None, 'norm': None}
    default_context = {'timeout': None, 'compute_per_structure': True, 'disable_external_wrap': False}
    # TODO: Fix unintended behaviour of 'timeout' being applied to per-structure
    # calculations individually. In reality, we want timeout = timeout/n_structures
    # when computing, but this needs the ability to pass through a new timeout throught
    # the cache wrappers. Later! For now, this just needs to be done by hand.

    def __init__(self, config, context={}):
        super().__init__(config, context=context)

        self.compute = self.compute_per_structure
        self.mbtr_gen = explicit_single_mbtr_4

        if self.compute_per_structure:
            cache_entries = 10000
        else:
            cache_entries = 25

        if self.cache_type == 'mem':
            self.mbtr_gen = memcached(self.mbtr_gen, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            disk_cached = diskcached(self.mbtr_gen, cache_location=self.cache_location, name='mbtr_4', min_duration=self.min_duration)
            self.mbtr_gen = memcached(disk_cached, max_entries=cache_entries)

        elif self.cache_type == 'disk':
            self.mbtr_gen = diskcached(self.mbtr_gen, cache_location=self.cache_location, name='mbtr_4', min_duration=self.min_duration)
