*******
Dataset
*******

Imports
=======

Found in: ``cmlkit.dataset``. For loading from a central location, you can use ``cmlkit.load_dataset`` (a convenience import for ``cmlkit.autoload.load_dataset``), which checks the :doc:`globals` for a path. Alternatively, the ``cmlkit.dataset.read`` function can be used to read files from disk.

Introduction
============

The ``Dataset`` class, alongside its subclass ``Subset`` provides a container for datasets, which for our purposes essentially consist of

* Geometries, either molecules or crystal structures
* Properties of these geometries (currently only scalar, but an extension to vector-valued properties is not too difficult)
* Metadata, first of all an ``id`` that is supposed to uniquely identify a given dataset.
  
Datasets are created from an array of elementary charges, ``z``, a ragged array of geometries ``r`` and, if crystals are treated, a ragged array of basis vectors ``b``. The first index in these arrays is assumed to correspond to a given structure. Additionally, properties are stores as dictionary, with the key being a short name, and the value an array with the values for a given structure.

Subsets can be created from datasets by passing an array of indices. They are intended to be persistent, for instance when setting aside a permanent test set.

Loading can happen either with the ``cmlkit.load_dataset`` method, or by calling ``Dataset.from_file``. 

Saving is achieved by with the ``save`` method.

Calling ``print(dataset)`` will print a detailed report on the dataset, which, crucially, includes recommendations for geometry function ranges... this info can also be accessed as ``dataset.info['geometry']``. (Note that the latter will not include safety margins etc.)

Calling ``__getitem__`` on a Dataset will return a ``View`` object that represents a dynamically-created subset of the data. This is intended for temporary use, for instance when looking at transient splits of a given dataset.

For additional details, please check the source code, or the automatically generated docs below. (Or the examples, which are not public yet.)

(Naming) Conventions
====================

Datasets are always saved with the extension ``.dat.npy``.

Some care has been taken to make sure that datasets can be identified and addressed. So here are the different bits of meta-data:

* ``name``: The name of the dataset, identical to the ``id`` for ``Datasets``, but not ``Subsets``
* ``id``: A (ideally) unique identifier. For Subsets, this is automatically generated as ``parentname-subsetname``.
* ``family``: A broader category identifier. At present, this is only used for the property converter.
* ``desc``: A short text description of the dataset

The default filename is the ``id``.

Internally, datasets are also hashed on creation, with some rudimentary sanity-checking.

Docs
====

Dataset
-------

.. autoclass:: cmlkit.Dataset
    :members:
    :undoc-members:


Subset
------

.. autoclass:: cmlkit.Subset