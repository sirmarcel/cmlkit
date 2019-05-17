import os
import numpy as np
import time
import ase
import ase.io
import subprocess
from pathlib import Path
import shutil

import cmlkit as cml

run_quippy = Path(__file__).parents[0] / 'run_quippy.py'


class Soap(cml.engine.Component):
    """SOAP descriptor as implemented in quippy"""

    kind = 'soap'

    default_context = {'timeout': None, 'cleanup': True, 'cache_type': None, 'min_duration': 30.0}

    def __init__(self, sigma, n_max, l_max, cutoff, elems=None, context={}):

        super().__init__(context=context)
        self.cache_type = self.context['cache_type']
        self.min_duration = self.context['min_duration']
        self.timeout = self.context['timeout']
        self.cleanup = self.context['cleanup']

        self.sigma = sigma
        self.n_max = n_max
        self.l_max = l_max
        self.cutoff = cutoff
        self.elems = elems

        self.computer = compute_soap
        cache_entries = 10

        if self.cache_type == 'mem':
            self.computer = cml.engine.memcached(self.computer, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            disk_cached = cml.engine.diskcached(self.computer, cache_location=cml.cache_location, name='soap', min_duration=self.min_duration)
            self.computer = cml.engine.memcached(disk_cached, max_entries=cache_entries)

        elif self.cache_type == 'disk':
            self.computer = cml.engine.diskcached(self.computer, cache_location=cml.cache_location, name='soap', min_duration=self.min_duration)

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {
            'sigma': self.sigma,
            'n_max': self.n_max,
            'l_max': self.l_max,
            'cutoff': self.cutoff,
            'elems': self.elems,
        }

    def compute(self, data):
        return self.computer(data,
                             self.sigma,
                             self.n_max,
                             self.l_max,
                             self.cutoff,
                             elems=self.elems,
                             timeout=self.timeout,
                             cleanup=self.cleanup,
                             )


def compute_soap(data, sigma, n_max, l_max, cutoff, elems=None, cleanup=True, timeout=None):
    # move this here so when caches are loaded no dependencies are needed;
    # so when running on a computer without them we can at least make use of cached stuff.
    check_dependencies()

    if elems is None:
        elems = data.info['elements']

    config = make_config(sigma, n_max, l_max, cutoff, elems)
    folder = prepare_task(data, config, elems)
    stdout, stderr = run_task(folder, timeout=timeout)
    result = read_result(folder)

    if cleanup:
        shutil.rmtree(folder)

    return result


def check_dependencies():
    assert cml.quippy_pythonpath is not None, f"Could not find $CML2_QUIPPY_PYTHONPATH, which is needed to compute SOAP."
    assert cml.quippy_python_exe is not None, f"Could not find $CML2_QUIPPY_PYTHON_EXE, which is needed to compute SOAP."


def make_config(sigma, n_max, l_max, cutoff, elems):
    species = ' '.join(map(str, elems))
    config = f"soap cutoff={cutoff} l_max={l_max} n_max={n_max} atom_sigma={sigma} n_Z={len(elems)} Z={{{species}}} n_species={len(elems)} species_Z={{{species}}}"
    return config


def prepare_task(data, config, elems):
    tid = 'soap_' + str(cml.engine.compute_hash(time.time() + np.random.rand()))
    folder = cml.scratch_location / Path(tid)
    folder.mkdir(parents=True)

    write_data(data, folder)

    with open(folder / 'config.txt', 'w+') as f:
        f.write(config)

    return folder


def write_data(data, folder):
    if data.b is None:
        a = [ase.Atoms(positions=data.r[i], numbers=data.z[i]) for i in range(data.n)]
    else:
        a = [ase.Atoms(positions=data.r[i], numbers=data.z[i], cell=data.b[i]) for i in range(data.n)]

    ase.io.write(str(folder / 'data.traj'), a, format='traj', parallel=False)


def run_task(folder, timeout=None):
    env = {
        'PYTHONPATH': cml.quippy_pythonpath,  # environment containing quippy and ase
        'HOME': os.environ['HOME']
    }

    python_exe = cml.quippy_python_exe  # python 2.7 executable
    scriptpath = run_quippy

    finished = subprocess.run(
        [python_exe, scriptpath],
        cwd=folder,
        timeout=timeout,
        # check=True,
        encoding="utf-8",
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env
    )

    if 'Error' in finished.stderr or 'Error' in finished.stdout:
        # TODO: make more useful errors
        raise Exception(f"SOAP did not terminate correctly. Here is stderr:\n{finished.stderr}\nHere is stdout:\n{finished.stdout}")


    return finished.stdout, finished.stderr


def read_result(folder):
    return np.load(folder / 'out.npy', fix_imports=True, encoding='bytes')
