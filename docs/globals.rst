*******
Globals
*******

``cmlkit`` supports the following global environment variables:

``CML_DATASET_PATH``: A collection of directories, formatted like the ``PATH`` environment variable. The ``cml.load_dataset`` method will look for datasets to load in these locations. This is particularly important for autotune. As last resort, the loader will just look in the current directory.

``CML_CACHE_SIZE``: An integer that determines the number of items held in memory caches. Currently, this is only relevant if the ``CachedMBTR`` representation is used, but this will be changed soon when caching is implemented for kernel matrices.

``CML_CACHE_LOC``: A path where on-disk caches are stored. Currently also only used by ``CachedMBTR``, but this might change in the future. Currently, the default is the current working directory (if a ``cache`` subdirectory exists in the current work directory, it is used instead).

In most cases, these do not have to be set. However, when ``autotune`` is used, it's usually a good idea to separate caches between different runs, and store datasets in a central location.