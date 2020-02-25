## Representations

This module deals with transforming molecules or crystals into an abstract vector representation, i.e. mapping to a space which is more amenable for regression. For many more details on this topic, you could read [this paper]( marcel.science/repbench )! 🤓

The base class for all representations is `Representation`, defined in `representation.py`. It takes care of some housekeeping, and will later also ensure that caching is implemented in a reasonable way. **This class also clearly defines the interface you have to implement for custom representations.**

The `mbtr` submodule is the interface to the "Many Body Tensor Representation" as implemented in [`qmmlpack`](https://gitlab.com/qmml/qmmlpack/-/tree/development). `soap` is the interface to [`quippy`](https://libatoms.github.io/QUIP/quippy.html) for computing the Smooth Overlap of Atomic Positions representations. `sf` is the interface to [`RuNNer`](https://gitlab.com/TheochemGoettingen/RuNNer) to compute Symmetry Functions. (For citations please see the main readme.)

If you do not have access to `quippy` or `ruNNer`, it is recommended to simply use the [`cscribe`](https://github.com/sirmarcel/cscribe) plugin which implements an interface for [`dscribe`](https://github.com/SINGROUP/dscribe). In tentative tests, `dscribe` performs at least as well as these reference implementations! The interface is largely the same for both SOAP and SF.