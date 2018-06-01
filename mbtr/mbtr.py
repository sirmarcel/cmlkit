import numpy as np
import qmmlpack as qmml
import cmlkit.inout as cmlio
import cmlkit.dataset as cmld
from cmlkit.mbtr.funcs import make_mbtrs
from cmlkit.utils.hashing import hash_sortable_dict, hash_arrays
from cmlkit import logger

version = 0.1  # currently not in use


class MBTR(object):
    """MBTR representation of a Dataset

    This class exists to provide a clean way to create,
    save and restore MBTR representations, which we
    regard as composed of a Dataset and a ModelSpec.

    Attributes:
        name: String; Name of this representation
        mbtr: Ndarray; Computed MBTRs
        spec: Dict, the mbtrs part of a ModelSpec
        spec_hash: Hash of the MBTR spec
        spec_name: String with name of ModelSpec used to generate this
        dataset_id: String; ID of original dataset
        version: Version of MBTR file, not in use
        hashes: Dict with various hashes that will be used for sanity-checking

    """

    def __init__(self, dataset, spec, name=None, restore_data=None):
        super(MBTR, self).__init__()


        self.version = version

        if restore_data is None:
            self.spec = spec.mbtrs

            if name is None:
                if isinstance(dataset, cmld.View):
                    self.name = dataset.id + '_view_n' + str(dataset.n)
                else:
                    self.name = dataset.id

                self.name += '_'
                self.name += spec.name
            else:
                self.name = name

            self.dataset_id = dataset.id
            self.spec_name = spec.name
            self.mbtr = self._make_mbtr(dataset, self.spec)

            self.hashes = {'spec': spec.hashes['mbtrs'],    # hash of the 'recipe' for the MBTR
                           'geom': dataset.hashes['geom'],  # hash of the data used to compute it
                           'mbtr': hash_arrays(self.mbtr)}  # hash of the result

        else:
            self.name = restore_data['name']
            self.hashes = restore_data['hashes']

            # Backwards compatibility
            if 'dataset_id' not in restore_data and 'dataset_name' in restore_data:
                logger.warn('Encountered deprecated MBTR file {}: missing dataset_id.'.format(self.name))
                self.dataset_id = 'NOTID_' + restore_data['dataset_name']

            else:
                self.dataset_id = restore_data['dataset_id']

            if 'spec_name' not in restore_data:
                logger.warn('Encountered deprecated MBTR file {}: missing spec_name.'.format(self.name))
                self.spec_name = 'NOTSPECNAME_' + self.name.split('_', 1)[1]

            else:
                self.spec_name = restore_data['spec_name']

            self.spec = restore_data['spec']
            self.mbtr = restore_data['mbtr']

    def __getitem__(self, idx):
        return MBTRView(self, idx)

    def _make_mbtr(self, dataset, spec):
        return make_mbtrs(dataset, spec)

    def save(self, directory='', filename=None):
        tosave = {
            'name': self.name,
            'dataset_id': self.dataset_id,
            'spec': self.spec,
            'spec_name': self.spec_name,
            'mbtr': self.mbtr,
            'version': self.version,
            'hashes': self.hashes
        }

        if filename is None:
            cmlio.save(directory + self.name + '.mbtr', tosave)
        else:
            cmlio.save(directory + filename + '.mbtr', tosave)

    @classmethod
    def from_file(cls, file):
        d = cmlio.read(file, ext=False)
        return cls(None, None, restore_data=d)

    @property
    def raw(self):
        return self.mbtr


class MBTRView(object):
    """View onto an MBTR

    This class is intended to be used when only parts of
    an MBTR need to be accessed
    """

    def __init__(self, mbtr, idx):
        super(MBTRView, self).__init__()
        self.parent_mbtr = mbtr
        self.idx = idx

    @property
    def raw(self):
        return self.parent_mbtr.raw[self.idx]

    @property
    def name(self):
        return 'view_' + self.parent_mbtr.name

    @property
    def dataset_id(self):
        return self.parent_mbtr.dataset_id
