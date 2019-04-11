import numpy as np
import time
from pathlib import Path
import shutil
import subprocess

import cmlkit as cml


class BasicSymmetryFunctions(cml2.engine.BaseComponent):
    """Baseline of atom-centred symmetry functions"""

    kind = 'bsf'

    default_context = {'timeout': None, 'cleanup': True}

    def __init__(self, params, dim, elems=None, context={}):
        if cml2.runner_path is None:
            raise RuntimeError(f"Could not find $CML2_RUNNER_PATH, which is needed to compute symmetry functions.")

        if dim > 500:
            raise ValueError(f"ruNNer only supports up to 500 function values. You requested {dim}.")

        super().__init__(context=context)
        self.cache_type = self.context['cache_type']
        self.min_duration = self.context['min_duration']
        self.timeout = self.context['timeout']
        self.cleanup = self.context['cleanup']

        self.params = params
        self.dim = dim
        self.elems = elems

        self.computer = compute_symmfs
        cache_entries = 10

        if self.cache_type == 'mem':
            self.computer = cml2.engine.memcached(self.computer, max_entries=cache_entries)

        elif self.cache_type == 'mem+disk':
            disk_cached = cml2.engine.diskcached(self.computer, cache_location=cml2.cache_location, name='bsf', min_duration=self.min_duration)
            self.computer = cml2.engine.memcached(disk_cached, max_entries=cache_entries)

        elif self.cache_type == 'disk':
            self.computer = cml2.engine.diskcached(self.computer, cache_location=cml2.cache_location, name='bsf', min_duration=self.min_duration)

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {
            'params': self.params,
            'dim': self.dim,
            'elems': self.elems,
        }

    def compute(self, data):
        return self.computer(data, self.params,
                             self.dim,
                             elems=self.elems,
                             timeout=self.timeout,
                             cleanup=self.cleanup,
                             )


class EmpiricalSymmetryFunctions(BasicSymmetryFunctions):
    """Atom-Centred Symmetry Functions with largely empirical parametrisation 

    The scheme is adapted from Gastegger et. al, J. Chem. Phys. 148, 241709 (2018).

    Syntax for parametrisation:
        ['type', [N, *other_params_of_symmf]]

        So for instance
        ['centered', [10, 2, 10.0]]

        will generate kind 2 symmetry functions spaced from 0 to the cutoff 10.0 with 10 points.

        In more complex situations, None needs to be given for auto-inferred parameters.
    """

    kind = 'esf'

    def __init__(self, params, dim=None, elems=None, context={}):
        defaults = {'universal': [], 'elemental': []}
        self.my_params = {**defaults, **params}

        universal = []
        for i, spec in enumerate(self.my_params['universal']):
            if spec is not None:  # so I can easily turn off certain functions
                if spec[0] == 'centered' or spec[0] == 'shifted':
                    universal += generate_universal_symmfs(*spec)
                else:
                    # no generation needed
                    universal += [make_universal_symmf(*spec)]

        if len(self.my_params['elemental']) > 0:
            raise ValueError(f"EmpiricalSymmetryFunctions currently doesn't support elemental symmfs.")

        if dim is None:
            if elems is None:
                raise ValueError(f"elems must be specified to automatically infer dim!")
            dim = infer_dim(universal, len(elems))

        super().__init__({'universal': universal, 'elemental': []}, dim, elems=elems, context=context)

    def _get_config(self):
        return {
            'params': self.my_params,
            'dim': self.dim,
            'elems': self.elems,
        }


def generate_universal_symmfs(kind, args):
    if kind == 'centered':
        return _centered(*args)
    elif kind == 'shifted':
        return _shifted(*args)
    else:
        raise ValueError(f"Don't know empirical scheme {kind}.")


def _centered(n, kind, cutoff, eta=None, mu=None, zeta=None, lambd=None,):
    r0 = 0.5
    rn = cutoff - 1.0
    delta = (rn - r0) / (float(n - 1))

    return [make_universal_symmf(kind=kind, cutoff=cutoff, eta=0.5 / (r0 + i * delta)**2, mu=0.0, zeta=zeta, lambd=lambd)
            for i in range(n)]


def _shifted(n, kind, cutoff, eta=None, mu=None, zeta=None, lambd=None,):
    r0 = 0.5
    rn = cutoff - 1.0
    delta = (rn - r0) / (float(n - 1))

    return [make_universal_symmf(kind=kind, cutoff=cutoff, eta=0.5 / (delta)**2, mu=r0 + i * delta, zeta=zeta, lambd=lambd)
            for i in range(n)]


def infer_dim(symmfs, n_elems):
    n_twobody = 0
    n_threebody = 0
    for symmf in symmfs:
        if symmf[0] == 2:
            n_twobody += 1
        if symmf[0] == 3:
            n_threebody += 1

    return int(n_twobody * n_elems + n_threebody * n_elems * (n_elems + 1) / 2)


def make_universal_symmf(kind, cutoff, eta=None, mu=None, zeta=None, lambd=None,):
    if kind == 2:
        assert eta is not None
        assert mu is not None
        return [2, eta, mu, cutoff]

    if kind == 3:
        assert eta is not None
        assert lambd is not None
        assert zeta is not None
        return [3, eta, lambd, zeta, cutoff]

    else:
        raise ValueError(f"Symmetry function kind {kind} is not yet implemented.")


def make_elemental_symmf(kind, cutoff, eta=None, mu=None, zeta=None, lambd=None, z1=None, z2=None, z3=None,):
    if kind == 2:
        assert eta is not None
        assert mu is not None
        assert z1 is not None
        assert z2 is not None
        return [z1, 2, z2, eta, mu, cutoff]

    if kind == 3:
        assert eta is not None
        assert lambd is not None
        assert zeta is not None
        assert z1 is not None
        assert z2 is not None
        assert z3 is not None

        return [z1, 3, z2, z3, eta, lambd, zeta, cutoff]

    else:
        raise ValueError(f"Symmetry function kind {kind} is not yet implemented.")


def compute_symmfs(data, params, dim, elems=None, timeout=None, cleanup=True):
    """Compute atom-centred symmetry functions

    Args:
        data: Dataset instance
        params: dict containing a key 'universal' with ruNNer-style
                specifications of symmetry functions that will be
                applied to each element (combination), and a key 'elemental'
                containing a list of ruNNer-style specifications
                of symmetry functions per element type/combination.
        dim: dimensionality of descriptor, the maximum number of symmetry
             function values per element (will be truncated silently!)
        elems: if not None, a list of elements (as charges) to consider,
               defaults to all elements found in data
        timeout: maximum number of seconds to wait for computations
    """

    folder = prepare_task(data, params, elems)
    # TODO: error handling
    out, err = run_task(folder, timeout=timeout)
    descriptors = read_result(data, folder, dim)
    # TODO: post-processing for weighting etc.
    if cleanup:
        shutil.rmtree(folder)

    return descriptors


def prepare_task(data, params, elems):
    tid = cml2.engine.compute_hash(time.time() + np.random.rand())

    folder = Path(cml2.scratch_location) / f"runner_{tid}"
    folder.mkdir(parents=True)  # will raise error if already exists!

    with open(folder / 'input.data', 'w+') as f:
        datafile = make_datafile(data)
        f.write(datafile)

    with open(folder / 'input.nn', 'w+') as f:
        infile = make_infile(data, params, elems)
        f.write(infile)

    return folder


def make_infile(data, params, elems=None):
    lines = []

    # boilerplate
    lines.append("nn_type_short 1")
    lines.append("runner_mode 1")
    lines.append("parallel_mode 0")
    lines.append("energy_threshold 100.0")
    lines.append("bond_threshold 0.0")
    lines.append("use_short_nn")
    lines.append("global_hidden_layers_short 1")
    lines.append("global_nodes_short 1")
    lines.append("global_activation_short t")
    lines.append("use_atom_charges")
    lines.append("test_fraction 0.0")
    lines.append("cutoff_type 1")

    # elements to consider
    if elems is None:
        elems = data.info['elements']

    symbol_elements = [cml2.charges_to_elements[elem] for elem in elems]
    lines.append(f"number_of_elements {len(symbol_elements)}")
    lines.append(f"elements {' '.join(symbol_elements)}")
    for elem in symbol_elements:
        lines.append(f"atom_energy {elem} 0 ")

    for symmf in params['universal']:
        lines.append("global_symfunction_short " + " ".join(map(str, symmf)))

    for symmf in params['elemental']:
        lines.append("symfunction_short " + " ".join(map(str, symmf)))

    return "\n".join(lines)


def make_datafile(data):
    lines = []

    for i in range(data.n):
        z = data.z[i]
        r = data.r[i]

        lines.append("begin")
        if data.b is not None:
            for v in data.b[i]:
                lines.append(f"lattice {v[0]} {v[1]} {v[2]}")

        for j in range(len(z)):
            rs = r[j]
            zs = z[j]
            lines.append(f"atom {rs[0]} {rs[1]} {rs[2]} {cml2.charges_to_elements[zs]} 0.0 0.0 0.0 0.0 0.0")

        lines.append("end")

    return "\n".join(lines)


def run_task(folder, timeout=None):

    # this will raise an error if ruNNer does not
    # terminate with 0 or times out
    finished = subprocess.run(
        [cml2.runner_path, str(folder / 'input.data'), str(folder / 'input.nn')],
        cwd=folder,
        timeout=timeout,
        check=True,
        encoding="utf-8",
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    if 'ERROR' in finished.stdout:
        # TODO: make more useful errors
        raise Exception(f"ruNNer did not terminate correctly. Here is stdout:\n{finished.stdout}")

    with open(folder / 'STDOUT', 'w+') as f:
        f.write(finished.stdout)
    with open(folder / 'STDERR', 'w+') as f:
        f.write(finished.stderr)

    return finished.stdout, finished.stderr


def read_result(data, folder, dim, elems=None):
    if elems is None:
        elems = data.info['elements']

    elems = list(elems)  # so we can use the index method later
    total_elems = len(elems)

    total_atoms = data.info['total_atoms']
    atoms_by_system = data.info['atoms_by_system']

    descriptors = []

    i_system = 0  # index of system
    i_atom = 0  # position in descriptor array
    with open(folder / 'function.data', 'r') as f:
        for line in f:
            split = line.split()
            if len(split) == 1:
                # begin of system block
                i_atom_in_system = 0
                descriptor = np.zeros(((atoms_by_system[i_system]), dim * total_elems), dtype=float)  # need to explicitly cast as float to force conversion from string
            elif split == ['0.0000000000', '0.0000000000', '0.0000000000', '0.0000000000']:
                # end of system block
                i_system += 1
                descriptors.append(descriptor)
            else:
                # an actual output line!

                # sanity check
                z = int(split[0])  # charge of the atom we're looking at
                assert z == data.z[i_system][i_atom_in_system], f"Encountered irregularity while parsing ruNNer output! check {folder}"

                symmf = split[1::]

                # since the structure of symmfs varies by element type, we need
                # to separate them in the descriptor. here we do this by placing
                # the symmfs belonging to one element into blocks (rest stays zero)
                offset = dim * elems.index(z)
                # this truncates values if dim < len(symmf)
                realdim = min(dim, len(symmf))

                descriptor[i_atom_in_system][offset:offset + realdim] = symmf[0:realdim]
                i_atom_in_system += 1
                i_atom += 1

    # more (temporary) sanity checks
    assert i_system == data.n, f"Encountered irregularity while parsing ruNNer output! check {folder}"
    assert len(descriptors) == data.n, f"Encountered irregularity while parsing ruNNer output! check {folder}"
    assert i_atom == total_atoms, f"Encountered irregularity while parsing ruNNer output! check {folder}"

    return np.array(descriptors, dtype='O')  # object ndarray for fancy indexing, it's really a list
