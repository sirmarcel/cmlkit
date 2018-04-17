import qmmltools.inout as qmtio
import pprint

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
    file, which is human-writeable, but the canonical
    form is a binary .npy file. We therefore support
    creating this class from yaml, but only allow saving
    as .npy.

    Attributes:
        name: Short name of model
        desc: Short description of model
        version: Version of format; not relevant now
        data: Dict specifying the data this model is built for
        mbtrs: Dict of dicts specifying the parameters of MBTRs
        krr: Dict of arguments for KRR

    """

    def __init__(self, name, desc, data, mbtrs, krr, version):
        super(ModelSpec, self).__init__()

        self.name = name
        self.desc = desc
        self.data = data

        mbtrs_with_defaults = {}

        for mbtr, params in mbtrs.items():
            mbtrs_with_defaults[mbtr] = {**mbtr_defaults, **params}

        self.mbtrs = mbtrs_with_defaults
        self.krr = krr
        self.version = version  # version of format

    @classmethod
    def from_dict(cls, d):
        defaults = {
            'version': 0.1,
        }

        d = {**defaults, **d}

        return cls(d['name'], d['desc'], d['data'], d['mbtrs'], d['krr'], d['version'])

    @classmethod
    def from_yaml(cls, filename):
        d = qmtio.read_yaml(filename)
        return cls.from_dict(d)

    @classmethod
    def from_file(cls, filename):
        d = qmtio.read(filename, ext=False)
        return cls.from_dict(d)

    def save(self, folder='', filename=None):
        to_save = {
            'name': self.name,
            'desc': self.desc,
            'version': self.version,
            'data': self.data,
            'mbtrs': self.mbtrs,
            'krr': self.krr
        }

        if filename is None:
            qmtio.save(folder + self.name + '.spec', to_save)
        else:
            qmtio.save(folder + filename + '.spec', to_save)

    @property
    def info(self):
        general = '###### {} ######\n'.format(self.name) + self.desc + '\n\n'
        data = '### DATA ###\n' + pprint.pformat(self.data) + '\n\n'
        krr = '### KRR ###\n' + pprint.pformat(self.krr) + '\n\n'
        mbtrs = '### MBTR ###\n' + pprint.pformat(self.mbtrs)
        return general + data + krr + mbtrs + '\n'
