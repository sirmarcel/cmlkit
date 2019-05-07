import numpy as np
from tqdm import tqdm

from cmlkit.engine import compute_hash
from infra import run_benchmark


def hash_1mb():
    a = np.random.rand(int(1 * 8 * 1024 ** 2 / 64))  # 1.0 MB of random data
    compute_hash(a)


def hash_1gb():
    a = np.random.rand(int(1 * 8 * 1024 ** 3 / 64))  # 1.0 GB of random data
    compute_hash(a)


def hash_complex_1mb():
    a = np.random.rand(int(1 * 8 * 1024 ** 2 / 64))  # 1.0 MB of random data
    to_hash = {"a": a, "b": 2 * a, "c": [1, 2, 3]}
    compute_hash(a)


def hash_complex_1gb():
    a = np.random.rand(int(1 * 8 * 1024 ** 3 / 64))  # 1.0 GB of random data
    to_hash = {"a": a, "b": 2 * a, "c": [1, 2, 3]}
    compute_hash(a)


reports = []
for f in tqdm([hash_1mb, hash_1gb, hash_complex_1mb, hash_complex_1gb]):
    reports.append(run_benchmark(f))

for r in reports:
    print(r)
