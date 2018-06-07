************
Installation
************

*Note*: Currently, ``cmlkit`` is under heavy development, so I wouldn't recommend bothering to copy it to ``site-packages`` or something like that. Just add it to ``$PYTHONPATH`` (see below.) Otherwise, ``python setup.py install`` should work, but will not currently install dependencies.

Dependencies
============

First, and crucially, you must have a current version of ``qmmlpack`` installed on your system. This document does not cover installing it. (Not yet!)

*Note*: If you use ``conda``, which is highly recommended, you can let it automatically install the dependencies (except ``qmmlpack``). If you want to do so, just skip to "Actual Installation".

The following (``conda``) packages are required:

* ``numpy``
* ``hyperopt``
* ``nose``
* ``pyyaml``

For automated hyperparameter tuning, you also need to install MongoDB, i.e. ``mongodb``

It is *highly* recommended to use Anaconda to manage and install these dependencies.

``cmlkit`` is being developed on Python 3.6. Any backwards compatibility is incidental, so please use a conda environment with 3.6.

For development, the following are required as well:

* ``sphinx``
* ``sphinxcontrib``

Actual Installation
===================

In your working directory (which I will call ``dir`` from now on), clone the project::
    
    cd dir
    git clone git@github.com:sirmarcel/cmlkit.git

This will create a sub-directory named ``cmlkit`` in ``dir``. Then ``cd`` into the directory.::

    cd cmlkit

If using ``conda``, now is the time to automatically install dependencies::
    
    conda env create -f environment.yml

This will create an environment named ``cmlkit`` with all required packages. Remember to activate this environment with ``source activate cmlkit``.

Now run the tests to check whether things are mostly okay::
    
    nosetests -v

This should generate a lot of output, and at the end it should say ``OK``. If that's not the case... e-mail me. :D

Setup
=====

In order for Python to be able to import ``cmlkit``, add ``dir`` to the ``PYTHONPATH``::

    export PYTHONPATH=$PYTHONPATH:'/path/to/dir/cmlkit/'


For the command-line interface to work, also add the following to ``PATH``::

    export PATH=$PATH:'/path/to/dir/cmlkit/cli'


``cmlkit`` employs a couple of environment variables, which are documented in :doc:`globals`. If you need to set any of these, please also do this.

All of the above should be put into your ``.profile/.bashrc/.zshrc`` so it is automatically executed.