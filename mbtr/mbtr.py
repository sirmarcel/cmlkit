import numpy as np
import qmmlpack as qmml
import qmmltools.inout as qmtio
import qmmltools.dataset as qmtd
from qmmltools.mbtr.funcs import make_mbtrs


class MBTR(object):
    """MBTR representation of a Dataset

    This class exists to provide a clean way to create,
    save and restore MBTR representations, which we
    regard as composed of a Dataset and a ModelSpec.

    Attributes:
        spec: ModelSpec (note that the krr key is not required)
        name: String; Name of this representation
        dataset_name: String; Name of original dataset
        mbtr: Ndarray; Computed MBTRs

    """
    def __init__(self, dataset, spec, name=None, restore_data=None):
        super(MBTR, self).__init__()

        if restore_data is None:
            self.spec = spec

            if name is None:
                if isinstance(dataset, qmtd.View):
                    self.name = dataset.name + '_view_n' + str(dataset.n)
                else:
                    self.name = dataset.name

                self.name += '_'
                self.name += spec.name
            else:
                self.name = name

            self.dataset_name = dataset.name
            self.mbtr = self._make_mbtr(dataset, self.spec.mbtrs)

        else:
            self.name = restore_data['name']
            self.dataset_name = restore_data['dataset_name']
            self.spec = restore_data['spec']
            self.mbtr = restore_data['mbtr']

    def __getitem__(self, idx):
        return self.mbtr[idx]

    def _make_mbtr(self, dataset, spec):
        return make_mbtrs(dataset, spec)

    def save(self, directory='', filename=None):
        tosave = {
            'name': self.name,
            'dataset_name': self.dataset_name,
            'spec': self.spec,
            'mbtr': self.mbtr
        }

        if filename is None:
            qmtio.save(directory + self.name + '.mbtr', tosave)
        else:
            qmtio.save(directory + filename + '.mbtr', tosave)

    @classmethod
    def from_file(cls, file):
        d = qmtio.read(file, ext=False)
        return cls(None, None, restore_data=d)

    @property
    def raw(self):
        return self.mbtr

