## SOAP 🧼

The Smooth Overlap of Atomic Positions representaton (SOAP) by [Bartók, Kondor, Csányi (2013)](https://doi.org/10.1103/PhysRevB.87.184115) is a local representation. Roughly speaking, it's an invariant constructed out of the expansion coefficients of a local atomic density in spherical harmonics and a radial basis.

This module implements the interface to the reference implementation [`quippy`](https://libatoms.github.io/QUIP/quippy.html). 

### Setup

**Note: This was written when `quippy` had no `python3` support. Presumably, this is no longer necessary -- I will check on this as soon as I can.**

For reasons detailed in `quippy_interface.py`, `cmlkit` cannot directly interface with `quippy`. Instead, we have to call a mini-CLI of `quippy` in a separate python 2.7 subprocess. In order for this to work, you need to:

- Install `quippy` in an isolated environment with python 2.7 (we recommend using a conda environment)
- Set the `$CML_QUIPPY_PYTHONPATH` to the `$PYTHONPATH` corresponding to this environment
- Set `$CML_QUIPPY_PYTHON_EXE` to point to the python 2.7 executable

This module will then write the data to disk, generate an input file, run `quippy`, and get the results. For the mechanics of this, check the `quippy`-prefixed files.

By default, `cmlkit` will create a `cml_scratch` directory in your current `pwd`, and store the intermediate input and output files there, cleaning up after itself. The location of this scratch directory can be controlled via the `$CML_SCRATCH` environment variable.

This is clearly insane, but it's the best we can do.

If the above is bothersome, I'd highly recommend using `dscribe` instead, via the[`cscribe`](https://github.com/sirmarcel/cscribe) plugin for `cmlkit`. In preliminary benchmarks, the SOAP implementation there is at least "as good" (in the sense of predictions errors and speed) as the one in `quippy`.


### Architecture

`soap.py` is the actual `Representation`, and explains the parameters in detail. 

`quippy_interface.py` prepares things for `quippy`. `quippy_execute.py` is the script executed in the python 2.7 environment and calls `quippy`.