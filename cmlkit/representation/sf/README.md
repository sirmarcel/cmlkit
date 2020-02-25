## Symmetry Functions

Symmetry functions are an atomic representation, consisting of an array of function values computed with different parametrisations. They were introduced by Jörg Behler in: [Behler (2011)](https://doi.org/10.1063/1.3553717).

In this module, we implement an interface with the reference implementation of SFs, the [`RuNNer`](https://gitlab.com/TheochemGoettingen/RuNNer) code. (Which also provides neural network potentials, but we don't use that here.) In addition, we provide an implementation of a common parametrisation scheme for SFs, in particular the one written up in [Gastegger *et al.* (2018)](https://doi.org/10.1063/1.5019667).

For pointers on how to use SFs, please check `sf.py`. For setup details, see below!

### Setup

You need to have `RuNNer` compiled on your machine. Then, set the `$CML_RUNNER_PATH` environment variable to point to the `RuNNer` executable.

By default, `cmlkit` will create a `cml_scratch` directory in your current `pwd`, and store the intermediate input and output files there, cleaning up after itself. (For details, check `runner.py` and `sf.py`) The location of this scratch directory can be controlled via the `$CML_SCRATCH` environment variable.

If you do not have access to `RuNNer`, do not despair. Use `dscribe` instead, via the[`cscribe`](https://github.com/sirmarcel/cscribe) plugin for `cmlkit`.

### Architecture

```
  +--sf.py--------------------------+
  |                                 |
  | "User facing" class with        |
  | standard config interface.      |
  |                                 |
  | Check here for details on       |
  | the parameters of SFs.			    |
  |                                 |
  | Supports parametrisation schemes|
  +---+-----------------------------+
      |
      | High-level config
      |  (can contain parametrisation directives)
      v
  +--config.py----------------------+
  |                                 |
  | Apply parametrisation schemes,  |
  | deduplicate and normalise,      |
  | generate "real" parameters.     |
  |                                 |
  +---+-----------------------------+
      |
      | Explicit canonical config
      | 
      v
  +--runner.py and friends----------+
  |                                 |
  | Interface with the RuNNer code. |
  | Generate input files, dispatch  |
  | shell calls to RuNNer, read in  |
  | results and process them.       |
  |                                 |
  +---+-------------------------+---+
      |                         ^
      | Input files in          | Output files,
      | a scratch dir.          | in same scratch dir.
      v                         |
  +-RuNNer----------------------+---+
  |                                 |
  | The actual RuNNer code!         |
  |                                 |
  +---------------------------------+
  
```

This slightly baroque structure is neccessary because by design, SFs are often used in a large set of slightly different parametrisations, each of which has a handful of parameters. This is incredibly unwieldy as a user, so the "outside-facing" interface allows you to use automatic parametrisation schemes, which are, in essence, shortcuts for entire sets of SFs. `config.py` simply expands these shortcuts and generates a more explicit config, which can easily be translated into the actual input file for `RuNNer`.

An additional advantage of doing things this way is that we can use the intermediate config as convenient input for alternative implementations of SFs, for instance in the `cscribe` plugin, without having to reimplement the parametrisation schemes elsewhere.
