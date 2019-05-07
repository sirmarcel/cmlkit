from cmlkit.engine import time_repeat


def run_benchmark(f, repeats=5):
    startstring = f"Running benchmark of {f.__name__} with {repeats} reps..."
    print(startstring)
    report = startstring + "\n"

    res = time_repeat(f, repeats)
    resstring = f"Mean, min, max: {res[1]:2f}, {res[2]:2f}, {res[3]:2f}."
    print(resstring)

    report += resstring + "\n"

    return report
