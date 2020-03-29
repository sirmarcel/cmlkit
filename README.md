# cmlkit 🐫🧰

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cmlkit.svg) [![PyPI](https://img.shields.io/pypi/v/cmlkit.svg)](https://pypi.org/project/cmlkit/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black) 

Publications: [`repbench`: Langer, Gößmann, Rupp (2020)](https://marcel.science/repbench)

Plugins: [`cscribe 🐫🖋️`](https://github.com/sirmarcel/cscribe) | [`mortimer 🎩⏰`](https://gitlab.com/sirmarcel/mortimer) | [`skrrt 🚗💨`](https://gitlab.com/sirmarcel/skrrt)

***

`cmlkit` is an extensible `python` package providing clean and concise infrastructure to specify, tune, and evaluate machine learning models for computational chemistry and condensed matter physics. Intended as a common foundation for more specialised systems, not a monolithic user-facing tool, it wants to help you build your own tools! ✨

*If you use this code in any scientific work, please mention it in the publication, cite [the paper](https://marcel.science/repbench) and let me know. Thanks! 🐫*

## What exactly is `cmlkit`?

[💡 A tutorial introduction to `cmlkit` courtesy of the NOMAD Analytics Toolkit 💡](https://www.nomad-coe.eu/index.php?page=bigdata-analyticstoolkit)

*Sidenote*: If you've come across this from outside the "ML for materials and chemistry" world, this will unfortunately be of limited use for you! However, if you're interested in ML infrastructure in general, please take a look at `engine` and `tune`, which are not specific to this domain and might be of interest.

### Features

- Reasonably clean, composable, modern codebase with little magic ✨

#### Representations

`cmlkit` provides a unified interface for:

- Many-Body Tensor Representation by [Huo, Rupp (2017)](https://arxiv.org/abs/1704.06439) (`qmmlpack` and `dscribe` implementation)
- Smooth Overlap of Atomic Positions representaton by [Bartók, Kondor, Csányi (2013)](https://doi.org/10.1103/PhysRevB.87.184115) (`quippy`‡ and `dscribe` implementations)
- Symmetry Functions representation by [Behler (2011)](https://doi.org/10.1063/1.3553717) (`RuNNer` and `dscribe` implementation), with a semi-automatic parametrisation scheme taken from [Gastegger et al. (2018)](https://doi.org/10.1063/1.5019667).

‡ The `quippy` interface was written for an older version that didn't support `python3`.

#### Regression methods

- Kernel Ridge Regression as implemented in [`qmmlpack`](https://gitlab.com/qmml/qmmlpack) (supporting both global and local/atomic representations)

#### Hyper-parameter tuning

- Robust multi-core support (i.e. it can automatically kill timed out external code, even if it ignores `SIGTERM`)
- No `mongodb` required
- Extensions to the `hyperopt` priors (uniform `log` grids)
- Resumable/recoverable runs backed by a readable, atomically written history of the optimisation (backed by [`son`](https://github.com/flokno/son))
- Search spaces can be defined entirely in text, i.e. they're easily writeable, portable and serialisable
- Possibility to implement multi-step optimisation (experimental at the moment)
- Extensible with custom loss functions or training loops

#### Various

- Automated loading of datasets by name
- Seamless conversion of properties into per-atom or per-system quantities. Models can do this automatically!
- Plugin system! ☢️ Isolate one-off nightmares! ☢️
- Canonical, stable hashes of models and datasets!
- Automatically train models and compute losses!

### But what... is it?

At its core, `cmlkit` defines a unified `dict`-based format to specify model components, which can be straightforwardly read and written as `yaml`. Model components are implemented as pure-ish functions, which is conceptually satisfying and opens the door to easy pipelining and caching. Using this format, `cmlkit` provides interfaces to many representations and a fast kernel ridge regression implementation.

Here is an example for a SOAP+KRR model:

```yaml
model:
  per: cell
  regression:
    krr:               # regression method: kernel ridge regression
      kernel:
        kernel_atomic: # soap is a local representation, so we use the appropriate kernel
          kernelf:
            gaussian:  # gaussian kernel
              ls: 80   # ... with length scale 80
      nl: 1.0e-07      # regularisation parameter
  representation:
    ds_soap:           # SOAP representation (dscribe implementation via plugin)
      cutoff: 3	
      elems: [8, 13, 31, 49]
      l_max: 8
      n_max: 2
      sigma: 0.5
```

Having a canonical model format allows `cmlkit` to provide a quite pleasant interface to `hyperopt`. The same mechanism *also* enables a simple plugin system, making `cmlkit` easily exensible, so you can isolate one-off task-specific code into separate projects without any problems, while making use of a solid, if opionated, foundation.

For a gentle, detailed tour please [check out the tutorial]( https://www.nomad-coe.eu/index.php?page=bigdata-analyticstoolkit ).

### Caveats 😬

Okay then, what are the rough parts?

- `cmlkit` is very inconvenient for interactive and non-automated use: Models cannot be saved and caching is not enabled yet, so all computations (representation, kernel matrices, etc.) must be re-run from scratch upon restart. This is not a problem during HP optimisation, as there the point is to try *different* models, but it is annoying for exploring a single model in detail. Fixing this is an *active* consideration, though! After all, the code is written with caching in mind.
- `cmlkit` is and will remain "scientific research software", i.e. it is prone to somewhat haphazard development practices and periods of hibernation. I'll do my best to avoid breaking changes and abandonement, but you know how it is!
- `cmlkit` is currently in an "alpha" state. While it's pretty stable and well-tested for some specific usecases (like writing a [large-scale benchmarking paper](https://marcel.science/repbench)), it's not tested for more everyday use. There's also some internal loose ends that need to be tied up.
- `cmlkit` is not particularly user friendly at the moment, and expects its users to be python developers. See below for notes on documentation! 😀

## Installation and friends

`cmlkit` is available via pip:

```
pip install cmlkit
```

You can also clone this repository! I'd suggest having a look into the codebase in any case, as there is currently no external documentation.

If you want to do any "real" work with `cmlkit`, you'll need to install [`qmmlpack`](https://gitlab.com/qmml/qmmlpack/-/tree/development) **on the development branch**. It's fairly straightforward!

***

In order to compute representations with `dscribe`, you should install the [`cscribe`](https://github.com/sirmarcel/cscribe) plugin:

```
pip install cscribe
```
You need to also export `CML_PLUGINS=cscribe`.

To setup the `quippy` and `RuNNer` interface please consult the readmes in `cmlkit/representation/soap` and `cmlkit/representation/sf`.

***

For details on environment variables and such things, please consult the readme in the `cmlkit` folder.

## "Frequently" Asked Questions

### Where is the documentation?

At the moment, I don't think it's feasible for me to maintain separate written docs, and I believe that purely auto-generated docs are basically a worse version of just looking at the formatted source on Github or in your text editor. So I *highly* encourage to take a look there!

Most submodules in `cmlkit` have their own `README.md` documenting what's going on in them, and all "outside facing" classes have extensive docstrings. I hope that's sufficient! Please feel free to file an issue if you have any questions.

### I don't work in computational chemistry/condensed matter physics. Should I care?

The short answer is regrettably probably no. 

However, I think the architecture of this library is quite neat, so maybe it can provide some marginally interesting reading. The `tune` component is very general and provides, in my opinion, a delightfully clean interface to `hyperopt`. The `engine` is also rather general and provides a nice way to serialise specific kinds of python objects to `yaml`.

### Why should I use this?

Well, maybe if you:

- need to use any of the libraries mentioned above, especially if you want to use them in the same project with the same infrastructure,
- are tired of plain `hyperopt`,
- would like to be able to save your model parameters in a readable format,
- think it's neat?

My goal with this is to make it slightly easier for you to build up your own infrastructure for studying models and applications in our field! If you're just starting out, just take a look around!

