import numpy as np
import qmmlpack as qmml
import qmmltools.inout as qmtio
import qmmltools.dataset as qmtd


def make_mbtrs(dataset, spec_mbtrs):

    return np.concatenate([single_mbtr(dataset.z, dataset.r, dataset.b, args) for k, args in spec_mbtrs.items()], axis=1)


def single_mbtr(z, r, b, args):

    kgwdcea = [args['k'], args['geomf'], args['weightf'], (args['distrf'], (args['broadening'],)), args['corrf'], args['eindexf'], args['aindexf']]

    mbtr = qmml.many_body_tensor(z, r, args['d'], kgwdcea, flatten=args['flatten'], elems=args['elems'], basis=b, acc=args['acc'])

    if args['norm'] is not None:
        for i in range(len(z)):
            mbtr[i] /= np.sum(mbtr[i])
        return mbtr * args['norm']
    else:
        return mbtr


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
            self.mbtr = make_mbtrs(dataset, self.spec.mbtrs)

        else:
            self.name = restore_data['name']
            self.dataset_name = restore_data['dataset_name']
            self.spec = restore_data['spec']
            self.mbtr = restore_data['mbtr']

    def __getitem__(self, idx):
        return self.mbtr[idx]

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
