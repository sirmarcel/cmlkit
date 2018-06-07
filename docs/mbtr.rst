****
MBTR
****

Imports
=======

Found in: ``cmlkit.reps.mbtr`` and ``cmlkit.reps.cached_mbtr``. It is recommended to use the convenience import ``cmlkit.MBTR``.

Introduction
============

The ``MBTR`` class is currently the only implemented representation of a dataset. The many-body tensor representation (see `the paper <https://arxiv.org/abs/1704.06439>`_) is a representation of the geometry of structures, see the paper for details. It is implemented in ``qmmlpack``, and wrapped here by ``cmlkit``.

In essence, the ``MBTR`` is just a combination of two objects: A ``Dataset`` and a ``ModelSpec``, the former supplying the geometries, and the latter a description of what exactly has to be computed. This is reflected in the constructor of the class, see below.

An additional class exists, the ``DiskAndMemCachedMBTR``, which adds thorough caching to the ``MBTR``, saving individual k-body terms to disk, and also keeping them in memory for fast retrieval. This infrastructure is under active consideration, so don't rely on it too much. Autotune makes heavy use of it, however.

Saving is again accomplished with the ``save`` class method. The filename is the name of the ``MBTR``, which is ``datasetid_modelspecname`` by default. 

Loading is handled by the ``from_file`` class method.

The ``MBTR`` class keeps track of the hashes of the ``Dataset`` and ``ModelSpec`` that produced it. The more high-level regression functions implemented in ``cmlkit.regression`` check these hashes against each other to make sure that everything is in order.


Docs
====

MBTR
-------

.. autoclass:: cmlkit.MBTR
    :members:
    :undoc-members:
