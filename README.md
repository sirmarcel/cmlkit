# cmlkit

*"a kit for camels"*

`cmlkit` is a Python package providing tools to build machine learning models with the many-body tensor representation (via `qmmlpack`) for condensed matter physics and quantum chemistry. 

Its main features are:
- Unified classes to create, save and load datasets, model specifications and computed MBTRs,
- Convenience functions for simple regressions tasks, and
- Automated hyperparameter tuning (via `hyperopt`)

It also (hopefully) provides a simple framework to experiment with the MBTR and collaborate, and reduces the amount of boilerplate code required to get things up and running.

WARNING: At present, `qmmlpack` is *not* publicly available, and this package depends on it. Sorry.

## Resources

The [documentation](https://cmlkit.readthedocs.io/en/latest/) contains a lot of information on `cmlkit`, in particular the [installation instructions](https://cmlkit.readthedocs.io/en/latest/install.html).

[This repository](https://github.com/sirmarcel/cmlkit-examples) contains some basic examples to get started.
