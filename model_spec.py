import qmmltools.inout as qmtio
import pprint

version = 0.1

mbtr_defaults = {
    'corrf': 'identity',
    'norm': None,
    'flatten': True,
    'elems': None,
    'acc': 0.001
}


class ModelSpec(object):
    """An MBTR+KRR model specification

    This class is a container for the specification of
    a KRR+MBTR model, consisting of some metadata, the
    parameters of the MBTRs and the parameters of the
    KRR model. It can be conveniently built from a yaml
    file, and saves itself to yaml.

    For legacy reasons, reading and writing from .npy is
    supported as well.

    Attributes:
        name: String with unique name of model
        desc: String with description of model
        data: Dict specifying the data this model is built for;
              should contain the key 'property' at a minimum, defining
              the property that it is intended to predict.
        mbtrs: Dict of dicts specifying the parameters of MBTRs; for
               details please refer to the (TODO) docs
        krr: Dict of arguments for KRR; check the (TODO) docs for details
        version: Version of format; can be safely ignored at present, but
                 will be used for backwards compatibility in the future

    """

    def __init__(self, name, desc, data, mbtrs, krr):
        super(ModelSpec, self).__init__()

        self.name = name
        self.desc = desc
        self.data = data

        mbtrs_with_defaults = {}

        for mbtr, params in mbtrs.items():
            mbtrs_with_defaults[mbtr] = {**mbtr_defaults, **params}

        self.mbtrs = mbtrs_with_defaults
        self.krr = krr

        self.version = version

    @classmethod
    def from_dict(cls, d):

        return cls(d['name'], d['desc'], d['data'], d['mbtrs'], d['krr'])

    @classmethod
    def from_yaml(cls, filename):
        """Read ModelSpec from .spec.yaml file."""

        d = qmtio.read_yaml(filename)
        return cls.from_dict(d)

    @classmethod
    def from_file(cls, filename):
        """Read ModelSpec from (legacy) .spec.npy file."""

        d = qmtio.read(filename, ext=False)
        return cls.from_dict(d)

    def save(self, folder='', filename=None):
        """Save this ModelSpec to YAML."""

        to_save = {
            'name': self.name,
            'desc': self.desc,
            'version': self.version,
            'data': self.data,
            'mbtrs': self.mbtrs,
            'krr': self.krr
        }

        if filename is None:
            qmtio.save_yaml(folder + self.name + '.spec', to_save)
        else:
            qmtio.save_yaml(folder + filename + '.spec', to_save)

    @property
    def info(self):
        general = '###### {} ######\n'.format(self.name) + self.desc + '\n\n'
        data = '### DATA ###\n' + pprint.pformat(self.data) + '\n\n'
        krr = '### KRR ###\n' + pprint.pformat(self.krr) + '\n\n'
        mbtrs = '### MBTR ###\n' + pprint.pformat(self.mbtrs)
        return general + data + krr + mbtrs + '\n'

    def print_info(self):
        """Print information about this ModelSpec."""

        print(self.info)
