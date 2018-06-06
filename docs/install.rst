************
Installation
************

Prerequisites
=============

First, and crucially, you must have a current version of ``qmmlpack`` installed on your system. This document does not cover installing it. (Not yet!)

Additionally, the following (anaconda) packages are required:

* ``hyperopt``
* ``numpy``
* ``nosetests``

For automated hyperparameter tuning, you also need to install MongoDB, i.e. ``mongodb``

It is *highly* recommended to use Anaconda to manage and install these dependencies.

``cmlkit`` is being developed on Python 3.6. Any backwards compatibility is incidental, so please use a conda environment with 3.6.

For development, the following are required as well:

* ``sphinx``

Actual Installation
===================

In your working directory (which I will call ``dir`` from now on), clone the project::
    
    cd dir
    git clone git@github.com:sirmarcel/cmlkit.git

This will create a sub-directory named ``cmlkit`` in ``dir``. Then ``cd`` into the directory and run the tests to check whether things are mostly okay::


    cd cmlkit
    nosetests -v

This should generate a lot of output, and at the end it should say ``OK``. If that's not the case... e-mail me. :D

Setup
=====

In order for Python to be able to import ``cmlkit``, add ``dir`` to the ``PYTHONPATH``::

    export PYTHONPATH=$PYTHONPATH:'/path/to/dir'


For the command-line interface to work, also add the following to ``PATH``::

    export PATH=$PATH:'/path/to/dir/cmlkit/cli'


``cmlkit`` employs a couple of environment variables, which are documented in :doc:`globals`. If you need to set any of these, please also do this.

All of the above should be put into your ``.profile/.bashrc/.zshrc`` so it is automatically executed.