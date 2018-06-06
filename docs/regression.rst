**********
Regression
**********

Imports
=======

The ``cmlkit.regression`` module

Introduction
============

This module offers some functions to quickly make predictions and compute losses, using the ``Dataset``/``MBTR``/``ModelSpec`` infrastructure. The workhorses here are the ``compute_loss`` and ``train_and_predict`` methods. These are built to be used with separate datasets (and optionally representations) for training and prediction. If one shared dataset is used, with different indices, the methods prefixed with ``idx_`` should be used.


Docs
====

.. automodule:: cmlkit.regression
   :members: