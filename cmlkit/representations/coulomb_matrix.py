import numpy as np
import qmmlpack as qmml
import cmlkit2 as cml2
from cmlkit2.engine import BaseComponent, memcached, diskcached


def compute_cm_minimal(z, r, unit='Angstroms', padding=False, flatten=True, sort=True):
    return qmml.coulomb_matrix(z, r, unit=unit, padding=padding, flatten=flatten, sort=sort)


class CoulombMatrixMinimal(BaseComponent):
    """A representation which is simply all distances 

    Like OnlyCoords this will not work for multiple molecules/structures,
    it is purely for conformations."""

    kind = 'cm_minimal'

    def __init__(self,
                 unit='Angstroms',
                 padding=False,
                 flatten=True,
                 sort=True,
                 context={},
                 ):
        super().__init__(context=context)
        self.cache_type = self.context['cache_type']
        self.min_duration = self.context['min_duration']

        self.unit = unit
        self.padding = padding
        self.flatten = flatten
        self.sort = sort

        self.computer = compute_cm_minimal
        cache_entries = 10

        if self.cache_type == 'mem':
            self.computer = memcached(self.computer, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            disk_cached = diskcached(self.computer, cache_location=cml2.cache_location, name='cmm', min_duration=self.min_duration)
            self.computer = memcached(disk_cached, max_entries=cache_entries)

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {
            'unit': self.unit,
            'padding': self.padding,
            'flatten': self.flatten,
            'sort': self.sort,
        }

    def compute(self, data):
        return self.computer(data.z, data.r, unit=self.unit, padding=self.padding, flatten=self.flatten, sort=self.sort)
