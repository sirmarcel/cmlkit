import numpy as np
from tqdm import tqdm

from cmlkit.engine import wrap_external
from infra import run_benchmark


def something(x):
    return x


wrapped = wrap_external(something)


def big_wrapped():
    a = np.random.rand(int(1 * 8 * 1024 ** 3 / 64))  # 1.0 GB of random data
    wrapped(a)


def big_unwrapped():
    a = np.random.rand(int(1 * 8 * 1024 ** 3 / 64))  # 1.0 GB of random data
    something(a)


reports = []
for f in tqdm([big_wrapped, big_unwrapped]):
    reports.append(run_benchmark(f))

for r in reports:
    print(r)
