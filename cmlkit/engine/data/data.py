import numpy as np
from pathlib import Path

from cmlkit.engine.config import Configurable
from cmlkit.engine.inout import normalize_extension


class Data(Configurable):

    # Data subclasses must provide "kind" and "version"
    # as class attributes

    def __init__(self, data=None, info=None, version=0, context={}):
        if data is None:
            self.data = {}
        else:
            self.data = data

        if info is None:
            self.info = {}
        else:
            self.info = info

    def _get_config(self):
        return {"data": self.data, "info": self.info, "version": self.version}

    def dump(self, path, protocol=1):
        assert protocol == 1, "Data only supports protocol 1 (.npz)"

        write_data_npz(path, self.kind, self.data, self.info, self.version)


def load_data(path):
    path = Path(path)

    if path.suffix == ".npz":
        return load_data_npz(path)
    else:
        raise ValueError


def load_data_npz(path):

    with np.load(path, allow_pickle=True) as file:

        kind = file["kind"].item()
        protocol = file["protocol"].item()
        assert protocol == 1, "npz data should be protocol 1"

        info = file["info"].item()

        data = {}
        for name, array in file.items():
            if name.split("/")[0] == "data":
                data[name.split("/")[1]] = array

        config = {kind: {"info": info, "data": data, "version": 1}}

    from cmlkit import from_config
    return from_config(config)


def write_data_npz(path, kind, data, info, version):
    kwds = {"kind": kind, "info": info, "protocol": 1, "version": version}

    for name, array in data.items():
        kwds[f"data/{name}"] = array

    np.savez(normalize_extension(path, ".npz"), **kwds)
