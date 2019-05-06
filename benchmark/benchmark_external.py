import numpy as np
from tqdm import tqdm

from cmlkit.engine import wrap_external, time_repeat


def something(x):
    return x


wrapped = wrap_external(something)


def big_wrapped():
    a = np.random.rand(int(1 * 8 * 1024 ** 3 / 64))  # 1.0 GB of random data
    wrapped(a)


def big_unwrapped():
    a = np.random.rand(int(1 * 8 * 1024 ** 3 / 64))  # 1.0 GB of random data
    something(a)


def run_benchmark(f, repeats=5):
    startstring = f"Running benchmark of {f.__name__} with {repeats} reps..."
    print(startstring)
    report = startstring + "\n"

    res = time_repeat(f, repeats)
    resstring = f"Mean, min, max: {res[1]:2f}, {res[2]:2f}, {res[3]:2f}."
    print(resstring)

    report += resstring + "\n"

    return report


reports = []
for f in tqdm([big_wrapped, big_unwrapped]):
    reports.append(run_benchmark(f))

for r in reports:
    print(r)
