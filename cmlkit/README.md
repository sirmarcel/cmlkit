# `cmlkit` Developer Readme 🐫🧰

Hello! Welcome to the ~real readme~. Or rather, the one with all the juicy technical details and stuff. 🤖

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
