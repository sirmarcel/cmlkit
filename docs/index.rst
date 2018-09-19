.. cmlkit documentation master file, created by
   sphinx-quickstart on Fri Jun  1 17:36:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

********************
cmlkit Documentation
********************

*"a kit for camels"*

``cmlkit`` is a Python package providing tools to build machine learning models with the many-body tensor representation (via ``qmmlpack``) for condensed matter physics and quantum chemistry. 

Its main features are:
- Unified classes to create, save and load datasets, model specifications and computed MBTRs,
- Convenience functions for simple regressions tasks, and
- Automated hyperparameter tuning (via ``hyperopt``)

It also (hopefully) provides a simple framework to experiment with the MBTR and collaborate, and reduces the amount of boilerplate code required to get things up and running.

WARNING: At present, ``qmmlpack`` is *not* publicly available, and this package depends on it. Sorry.

Credit
======

If you use this code in any scientific work, please mention it in the publication and let me know. Thanks!

Contents
========

Welcome to the documentation. You might be interested in reading the :doc:`install` instructions first, and then read about the main classes: :doc:`dataset`, :doc:`model_spec` and :doc:`mbtr`. Functions for basic regression tasks are found in :doc:`regression`.

The :doc:`autotune` page contains all information needed to run automated hyperparameter searches. Some things will be made more clear by looking at :doc:`conventions` and :doc:`globals`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   dataset
   model_spec
   mbtr
   regression
   autotune
   autoload
   indices
   globals
   conventions
   losses
   grids


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
