## Many-Body Tensor Representation

The [Many-Body Tensor Representation (Huo, Rupp (2017))](https://arxiv.org/abs/1704.06439) is essentially a broadened histogram of the values of a *k*-body "geometry function", with contributions weighted by a "weighting functions". It is a *global* representation, i.e. it represents entire systems at once.

One specialty of the MBTR is that in practical applications, one typically uses a combination of different MBTRs with different parameters and/or values of *k*. This module is focused on implementing *individual* MBTRs, the concatenation and combination is taken care of by the `Composed` representation class found in `cmlkit.representation`.

`mbtr.py` implements the `MBTR` base class and the common interface for all of them. `mbtrs.py` implements the different `k`-body MBTRs, and gives a clear overview of which parameters are supported for each body order. `norm.py` implements an *extension* of the MBTR, the ability to normalise the MBTR in various ways. (Hat tip to `dscribe` for pointing out why this is important!)