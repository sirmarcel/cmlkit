import time
import qmmlpack as qmml
from qmmlpack.experimental import recursive_matrix_map
import numpy as np
from qmmlpack.experimental import ExtensiveKernelRidgeRegression

from cmlkit import logger, cache_location
from ..helpers import convert_sequence
from ..engine import *  # TODO TRY TO FIX THIS MESS LMAO


def get_kernelf(name):
    if callable(name):
        return name
    else:
        try:
            return getattr(qmml, 'kernel_' + name)
        except AttributeError:
            raise NotImplementedError("Kernel named {} is not implemented.".format(name))


class KRR(BaseComponent):
    """Kernel Ridge Regression via qmmlpack"""

    kind = "krr"

    needs_incidence = False
    default_context = {'print_timings': False}

    def __init__(self, nl, kernel, centering=False, context={}):
        super().__init__(context=context)

        self.nl = nl
        self.kernel = convert_sequence(kernel)
        self.centering = centering

        self.kernelf = get_kernelf(self.kernel[0])
        self.kernel_theta = self.kernel[1][0]

        self.print_timings = self.context['print_timings']

        self.cache_type = self.context['cache_type']
        self.cache_location = cache_location
        self.min_duration = self.context['min_duration']

        if self.cache_type == 'mem':
            self.kernelf = memcached(self.kernelf, max_entries=100)

        elif self.cache_type == 'mem+disk':
            disk_cached = diskcached(self.kernelf, cache_location=self.cache_location, name='kernel_' + self.kernel[0], min_duration=self.min_duration)
            self.kernelf = memcached(disk_cached, max_entries=100)

        elif self.cache_type == 'disk':
            self.kernelf = diskcached(self.kernelf, cache_location=self.cache_location, name='kernel_' + self.kernel[0], min_duration=self.min_duration)

        self.is_trained = False

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {'nl': self.nl, 'kernel': self.kernel, 'centering': self.centering}

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

        start = time.time()
        kernel_train = self.kernel_self(x)
        if self.print_timings:
            logger.info(f"Computed kernel_train in {time.time()-start:.2f}s.")

        start = time.time()
        self.krr = qmml.KernelRidgeRegression(kernel_train,
                                              y,
                                              theta=(self.nl,),
                                              centering=self.centering)

        if self.print_timings:
            logger.info(f"Trained KRR in {time.time()-start:.2f}s.")

        self.is_trained = True
        self.kernel_train = self.krr.kernel_matrix  # qmml internally copies the kernel matrix, no reason to keep it

    def predict(self, x):
        if self.is_trained is False:
            raise Exception('KRR must be trained before prediction!')

        start = time.time()
        kernel_pred = self.kernel_other(x)
        if self.print_timings:
            logger.info(f"Computed kernel_pred in {time.time()-start:.2f}s.")

        # prediction is typically very fast, so we don't bother timing it
        return self.krr(kernel_pred)

    def cv_train_and_predict(self, x, y, idx):
        start = time.time()
        kernel = self.kernel_self(x)
        if self.print_timings:
            logger.info(f"Computed overall kernel for cv in {time.time()-start:.2f}s.")

        start = time.time()
        results = []
        for train, pred in idx:
            kernel_train = kernel[np.ix_(train, train)]
            kernel_pred = kernel[np.ix_(train, pred)]

            krr = qmml.KernelRidgeRegression(kernel_train,
                                             y[train],
                                             theta=(self.nl,),
                                             centering=self.centering)

            results.append(krr(kernel_pred))

        if self.print_timings:
            logger.info(f"Trained and predicted over {len(idx)} splits in {time.time()-start:.2f}s.")

        return results

    def kernel_self(self, x):
        if self.print_timings:
            logger.info(f"   (dim={len(x[0])} per structure)")
        return self.kernelf(x, theta=self.kernel_theta)

    def kernel_other(self, x):
        if self.print_timings:
            logger.info(f"   (dim={len(x[0])} per structure)")
        return self.kernelf(self.x_train, z=x, theta=self.kernel_theta)


class ExtensiveKRR(KRR):
    """Implements KRR for atomic descriptors rather than structural ones"""

    needs_incidence = True
    kind = 'ekrr'

    default_context = {'max_size': 800, 'print_timings': False}

    def __init__(self, nl, kernel, centering=False, context={}):
        super().__init__(nl, kernel, centering=centering, context=context)
        self.max_size = self.context['max_size']

        self.kernelf = get_kernelf(self.kernel[0])  # makes no sense to cache this
        self.kernel_theta = self.kernel[1][0]

        if self.cache_type == 'disk':
            self.atomic_kernel_self = diskcached(atomic_kernel_self, cache_location, name='atomic_kernel_self')
            self.atomic_kernel_other = diskcached(atomic_kernel_other, cache_location, name='atomic_kernel_other')

        else:
            self.atomic_kernel_self = atomic_kernel_self
            self.atomic_kernel_other = atomic_kernel_other


    def kernel_self(self, x):
        if self.print_timings:
            logger.info(f"   (dim={len(x[0][0])} per atom)")
        return self.atomic_kernel_self(self.kernelf, self.kernel_theta, x, max_size=self.max_size)

    def kernel_other(self, x):
        if self.print_timings:
            logger.info(f"   (dim={len(x[0][0])} per atom)")
        return self.atomic_kernel_other(self.kernelf, self.kernel_theta, self.x_train, x, max_size=self.max_size)


def atomic_kernel_self(kernelf, theta, descriptors, max_size=800):
    """Compute the kernel in kernelf with atomic descriptors for one set of structures

    Assumes atomic descriptors of fixed length dim. Each structure
    has n_atoms which vary across structures.

    Args:
        kernelf: Callable following the qmmlpack scheme for kernel functions
        theta: Hyperparameters for kernelf
        descriptors: List (len = n_structures) of ndarrays with the atomic
                     descriptors along axis 1, i.e. n_atoms x dim arrays

    """

    return _atomic_kernel(kernelf, theta, descriptors, descriptors, symmetric=True, max_size=max_size)


def atomic_kernel_other(kernelf, theta, descriptors, other_descriptors, max_size=800):
    """Compute the kernel in kernelf with atomic descriptors between two sets of structures

    Assumes atomic descriptors of fixed length dim. Each structure
    has n_atoms which vary across structures.

    Args:
        kernelf: Callable following the qmmlpack scheme for kernel functions
        theta: Hyperparameters for kernelf
        descriptors: List (len = n_structures) of ndarrays with the atomic
                     descriptors along axis 1, i.e. n_atoms x dim arrays
        other_descriptors: second descriptors

    """

    return _atomic_kernel(kernelf, theta, descriptors, other_descriptors, symmetric=False, max_size=max_size)


def _atomic_kernel(kernelf, theta, descriptors, other_descriptors, symmetric=False, max_size=800):

    def f(range_a, range_b):
        start_a, stop_a = range_a
        start_b, stop_b = range_b

        d_a = descriptors[start_a:stop_a]
        d_b = other_descriptors[start_b:stop_b]
        o_a = _get_offsets(d_a)
        o_b = _get_offsets(d_b)
        x_a = np.concatenate(d_a, axis=0)
        x_b = np.concatenate(d_b, axis=0)

        k = kernelf(x_a, z=x_b, theta=theta)

        return qmml.partial_sum_matrix_reduce(k, o_a, indc=o_b)

    return recursive_matrix_map(f, (len(descriptors), len(other_descriptors)), max_size=max_size, out=None, symmetric=symmetric)


def _get_offsets(descriptors):
    counts = np.array([len(s) for s in descriptors], dtype=int)  # obtain n_atoms per structure

    # offsets: indices of beginning and end of descriptors belonging to structures
    offsets = np.zeros(len(counts) + 1, dtype=np.int_)
    offsets[1::] = np.cumsum(counts)  # -> offsets = [0, n_atoms_1, n_atoms_1 + n_atoms_2, ...]

    return offsets
