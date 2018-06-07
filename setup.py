from setuptools import setup

with open('cmlkit/__about__.py') as f:
    exec(f.read())

setup(
    name="cmlkit",
    version=__version__,
    packages=['cmlkit',
              'cmlkit.autotune',
              'cmlkit.reps',
              'cmlkit.utils'],
)
