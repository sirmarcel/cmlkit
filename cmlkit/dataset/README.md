In `cmlkit`, structures and their properties are stored
in objects of the `Dataset` class, which provides a neat
package to contain all the related arrays that make up
a set of structures, and a number of methods to handle them.

Datasets are expected to be placed in some user-defined repository,
to which the `$CML_DATASET_PATH`, a `$PATH`-like set of directories
should point. Out of this dataset path, the `load_dataset` then
tries to find the requested dataset.

Please note that this submodule is in terrible shape.
