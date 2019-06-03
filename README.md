# cmlkit

*"🐫 a kit for camels 🧰"*

`cmlkit` provides a clean and concise way to specify, tune, and evaluate machine learning models for computational chemistry and condensed matter physics, particular for atomistic predictions.

WARNINGS: 
- `cmlkit` depends on [`qmmlpack`](https://gitlab.com/qmml/qmmlpack), which is *not* yet publicly available.
- This is a "scientific code", i.e. development occurs infrequently and somewhat haphazardly. I'll try to not make breaking changes too often, and never in minor versions.

If you use this code in any scientific work, please mention it in the publication and let me know. Thanks! 🐫

## What is `cmlkit`? 🐫🧰

At its core, `cmlkit` defines a unified `dict`-based format to specify model components, which can be straightforwardly read and written as `yaml`. It provides interfaces to implementations of popular methods in its domain using this format.

On this basis, it then implements parallel hyperparameter optimisation (using `hyperopt` as backend), and provides tools to train models, make predictions, and evaluate those predictions. It is intended to be extensible and flexible enough for the demands of research. It is also "high-performance computing compatible", i.e. it can run in computing environments straight from the 90s. 🤓

### Compatibility

At the moment, there are interfaces for:

Representations:
- Many-Body Tensor Representation (MBTR) (Huo, Rupp, arXiv 1704.06439 (2017)) (`qmmlpack` interface)
- Smooth Overlap of Atomic Positions (SOAP) representaton (Bartok, Kondor, Csanyi, PRB 87, 184115 (2013)) (`quippy` interface)
- Symmetry Functions (SF) representation (Behler, JCP 134, 074106 (2011)) (`RuNNer` interface)

Regression methods:
- Kernel Ridge Regression (KRR) as implemented in [`qmmlpack`](https://gitlab.com/qmml/qmmlpack)

### Features

The hyperparameter optimisation boasts:
- Robust multi-core support (i.e. it can automatically kill timed out external code, even if it ignores `SIGKILL`)
- No `mongodb` required (important for *cough* certain comuting environments *cough*)
- Extensions to the `hyperopt` spaces (`log` grids)
- Possibility to implement multi-step optimisation (experimental at the moment)
- Resumable/recoverable runs backed by a readable, atomically written history of the optimisation (backed by [`son`](https://github.com/flokno/son))
- Search spaces can be defined entirely in text, i.e. they're easily writeable, portable and serialisable

## Frequently Asked Questions

(They are not actually frequently asked.)

### I don't work in computational chemsitry/condensed matter physics. Should I care?

The short answer is regrettably probably no. 

However, I think the architecture of this library is quite neat, so maybe it can provide some marginally interesting reading. The `tune` component is very general and provides, in my opinion, a delightfully clean interface to `hyperopt`.

### Why should I use this?

If you need to use any of the libraries mentioned above it might be more convenient. If you need to do hyperparameter optimisation and are tired of plain `hyperopt` it might be useful.
