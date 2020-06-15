# `cmlkit` Developer Readme 🐫🧰

Hello! Welcome to the *real readme*. Or rather, the one with all the juicy technical details and stuff. 🤖

## What's in this package?

Here is a brief overview over what's going on here... these are the bits in here that you should look at first.

- `engine`: Domain-independent heavy lifting. Deals with the basic `Component` class, its de/serialisation, i/o, hashing, ...
- `tune`: The largely self-sufficient `hyperopt` interface and related tooling
- `representation`: Representations of molecules and crystals
- `regression`: Regression methods, currently only kernel ridge regression
- `env.py`: Look into this to learn everything about the shell environment variables you can set.

This can wait a bit:

- `dataset`: The `Dataset` class stores, well, datasets, but it is very creaky and up for rewriting any day now.
- `evaluation`: Defines the interface for evaluating models, implements some basics, and provides loss functions.
- `utility`: Grab bag of stuff that didn't fit anywhere else.
- `model.py`: The `Model` class! Didn't fit anywhere else.
- `exceptions.py`: Our custom exceptions, which are barely used.

## Plugin system 🧩

The plugin system is very simple:

- Write a python package, let's say `drmdrkit`
- Classes in that Python package must inherit from `cmlkit.engine.Component`
- At the module level, you must provide a variable `components` containing a list of those classes
- Finally, to use these classes, export `CML_PLUGINS=drmdrkit`

If all of this is the case, `cmlkit.from_config()` will be able to de-serialise your custom objects with absolutely zero problems.

(For an example of how this works in practice, take a look into the `representation` submodule, which is basically a whole suite of plugins!)

## Testing

`cmlkit` uses plain old `unittest`, and expects you to use `nose` as test runner. The tests need `qmmlpack` to run.

## Roadmap

The rough plan for the future of `cmlkit` is:

- Transition `Dataset` to a `Data` subclass: 
	- Native support for "atom ragged" arrays (i.e. store linearised form and supply lookup functionality)
	- Consider splitting "geometry data" from "property data" (to avoid awkwardness around `geom_hash` vs `hash`)

Things that would be nice, but will take some time

- Make it possible to save trained `Models` (caching should take most of the pain out of retraining)
- Possibly implement an `sklearn` interface as plugin (`cmlsci`)
- Add real support for predictive variances, and models that give uncertainties in general

## Development practice

The goal with `cmlkit` is to strike a reasonable balance between proper software development and the constraints of research and `#phdlife`. 

So at the moment, development is undertaken in a not particularly rigorous form:

- Development occurs largely in the `develop-2.0` branch. Larger features are typically built in separate branches, but not always. More rigoorous workflows will be enforced if `cmlkit` ever attracts more than 1 developer. 😅
- Release versions (even alpha versions) are tagged and published to pypi. Release versions should be functional, i.e. the tests must run.
- Upon tagging a release version, the development branch is merged into `master`. So `master` is always at a release version.

I'm doing my best to avoid making large-scale breaking changes to the core parts of `cmlkit` (i.e. `engine`). If I do, these will typically (at least after it's out of alpha) occur only in major versions.

Tests are strongly encouraged, but when in doubt, an integration or smoke test is better than nothing. There is a fair bit of functionality that is extremely tedious to test and I don't have the time to figure it out, but will cause the entire test suite to explode if it fails.

All code is formatted with `black`. Google-style docstrings are encouraged. All modules should have a `README.md` file explaining what's going on, pointing out overall architectural choices and reasoning. Nitty-gritty details can be purely documented in code, that's fine. Code is written in American English, all documentation and non-code text in British English.

Dependency management is done using `poetry`, for better or for worse. There are no plans to support `conda` at present.

At the moment, this code is developed by and for one person. If you are seriously considering working with `cmlkit`, please do absolutely not hesitate to get in touch.
