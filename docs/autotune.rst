********
Autotune
********

``cmlkit.autotune`` is the sub-module in charge of automated hyperparameter tuning for MBTR+KRR models. It's built on ``hyperopt`` (see the `repo <https://github.com/hyperopt/hyperopt>`_), which uses Tree-Structured Parzen Estimators to probabilistically optimise over the awkward search spaces arising when testing parameter combinations. ``hyperopt``, and consequently ``autotune``, also supports *parallelisation* with Mongo DB. In the following, I will give a brief overview on how to use ``autotune`` to optimise a model.

Overview
========

In essence, the input for ``hyperopt`` is a (nested, tree-shaped) dictionary of parameters, where the parameters to be optimised are replaced by probability distributions, representing an (intuitive) prior on their values. Crucially, this tree can contain nested expressions, so for instance we can express the fact that choosing one particular geometry function in the MBTR also requires us to choose a broadening, which is entirely unrelated to the one chosen for an unrelated geometry function. 

In technical terms, the priors are expressed by special functions, for instance ``hp.choice('name_of_param', ['option_1', 'option_2'])``, representing a uniformly random choice between two options. Continuous functions also exist, but are not usually used here -- caches are much easier to fill if only discrete choices are available. A list can be found in section 2.1 of `this wiki page <https://github.com/hyperopt/hyperopt/wiki/FMin>`_. 

In addition to the stochastic search with ``hyperopt``, there is also the option of using a local log grid search as implemented in ``qmmlpack``. This is only suitable for continuous numerical parameters, and only for ones that are relatively cheap to evaluate.

In ``autotune``, one big dictionary is used for both, the specification of the search space *and* the configuration of the run itself. This dictionary can be passed directly into it, provided the necessary ``hyperopt`` functions have been created, or passed as a ``yaml`` file. We will focus on the latter use case, since it is much more suitable to remote deployment.

Config
======

The config file is structure as follows:

.. code-block:: yaml

    name: # name of this run
    project: # if this is part of a larger project, name of project
    desc: # description
    data:
      id: # id of dataset for optimising on (cross-validation will be used)
      property: # property to predict
    config:
      # autotune config,
      # see below
     
    spec:
      # essentially the same as a ModelSpec,
      # but with special tags for hyperopt
      # see below

Here is an example of a config ``yaml`` file, excluding the ``spec`` block. Everything not commented out is compulsory; entries commented out are optional, and I've included the defaults.

.. code-block:: yaml

    name: this_run # name of this run
    desc: just an example # description
    # project: default # if this is part of a larger project, name of project
    data:
      id: id # id of dataset for optimising on (cross-validation will be used)
      property: prop # property to predict
      # id-train: id # if no CV is used, this is the training set
      # id-test: id # if no CV is used, this is the test set to compute losses
      ## if id-train and id-test and id are all specified, autotune will assumed that id-* are sub-sets of id,
      ## and compute everything on the parent dataset, using the indices in id-* for training and test
    config:
      n_calls: 100
      # cv: [random, 3, 1500, 500]  # cross-validation config; see below
      # loss: rmse  # string specifying the loss to optimise, taken directly from cmlkit.losses
      # n_cands: 2  # number of candidate models to save (ranked by loss)
      # loglevel: INFO  # amount of logging, DEBUG prints a lot of internal stuff
      # timeout: None # if set, number of seconds after which a trial is aborted
      # parallel: false # if true, parallel mode with MongoDB will be engaged
      # if parallel is true, the following defaults are also applied:
      # db_name: project_name
      # db_port: 1234
      # db_ip: 127.0.0.1
      # lgs: # local grid search config; see below
      #   maxevals: 25
      #   resolution: 0.0001
    spec:
      # See below

If a config file is passed to autotune, the following transformations are applied:

* Losses are replaced with the appropriate functions (see :doc:`losses` for a list)
* Grid specifications are replaced with the actual grids (see below)
* Hyperopt specifications are replaced with hyperopt functions (see below)
* Local grid search directives are NOT changed, this is done during objective evaluation (so you can ``hyperopt`` the parameters of ``lgs``...)

Cross Validation
----------------
    
Currently, the only supported CV method is random cross-validation, with the following syntax: ``[random, n_cv, n_train, n_test]``. This means that ``n_cv`` times, the dataset is *randomly* split into ``n_train`` structures to train and ``n_test`` structures to predict. The loss is then computed for each, and averaged over the repetitions.

Otherwise, you just specify a training and test set, and the optimisation is done over these sets.

Local Grid Search
-----------------

In config, some overall parameters of the local grid search can be set.

For the syntax of lgs-optimised variables, see below.

Spec/Search space
=================

The tree-structured search space is defined in the ``spec`` block of the config file. Here is an example, with an MBTR that only has one body terms.

.. code-block:: yaml

    spec:
      data:
        property: fecreal
      mbtrs:
        mbtr_1:
          k: 1
          d: [-2.4, 1.76, 30]
          geomf: 'count'
          weightf: 
            - 'hp_choice'
            - 'mbtr1_weightf'
            - 
              - '1/count'
              - 'unity'
              - '1/identity'
              - 'identity_root'
              - 'identity'
              - 'identity^2'
              - ['exp_-1/identity', ['hp_choice', 'mbtr_1_wf_ls_1', ['gr_medium']]]
              - ['exp_-1/identity^2', ['hp_choice', 'mbtr_1_wf_ls_2', ['gr_medium']]]
          distrf: ['normal', ['hp_choice', 'mbtr_1_ls', ['gr_medium']]]
          eindexf: 'full'
          aindexf: 'full'
          norm:
            - 'hp_choice'
            - 'mbtr_1_norm'
            - [null, 0.1, 1.0, 5.0, 10.0]
      krr:
        kernelf: ['gaussian',  ['hp_choice', 'krr_ls', ['gr_log2', -20, 20, 41]]]
        nl: ['hp_choice', 'krr_nl', ['gr_medium']]
        centering: false

As you can see, the structure is identical to the one used in :doc:`model_spec`, with the addition of ``hp_`` and ``gr_`` in some places. This might look weird at first, but they follow a simple pattern: They are lists, shaped as ``['module_function', arg1, arg2]`` (following the pattern described in :doc:`conventions`). Two modules currently exist: ``hp_`` meaning ``hyperopt`` and ``gr_`` meaning grid. Let's look at these in detail.

Hyperopt Specifications
-----------------------

In principle, all ``hyperopt`` functions are supported, but in practice, we currently only use ``hp_choice`` (it might be worth investigating this in more detail). For ``hp_choice``, the first argument has to be a string uniquely identifying the parameter to be chosen. The second argument has to be a list of possible options. In the case of numerical parameters, it is tedious to specify these lists by hand, and so the ``grid`` module exists. However, we are *not* restricted to numerical parameters -- for instance, the choices could also be made between full dictionaries describing different MBTRs.

Also please note that the choices can be *nested* and *conditional*. For instance, in the above, some choices of ``weightf`` require an additional parameter, which is also chosen with ``hp_choice``. The same principle applies when whole MBTR configurations are put into ``hp_choice``. In such cases, some care has to be taken to keep the labels of parameters unique.

Grid Specifications
-------------------

Prefixed with ``gr_``. A few convenience shortcuts have been implemented, for instance ``gr_medium`` is a grid evenly spaced in log2 space from -18 to +20 in steps of 1. These shortcuts currently require the use of a one-item list, which is somewhat awkward, this will be changed at some point. 

``gr_log2`` generates a log2 grid. A full list can be found in :doc:`grids`.

In summary, the statement ``['hp_choice', 'krr_ls', ['gr_log2', -20, 20, 41]]`` will, at runtime, be replaced with a random pick of 2^-20, 2^-19, ... 2^19, 2^20. 

Local grid search specification
-------------------------------

Marked by ``lgs``, followed by a list following the syntax used in ``qmmlpack`` for a single variable, for instance:

``['lgs', [start_exponent, priority, step_size, min_exponent, max_exponent]]``

Note that the search is performed on a *log* grid with base 2. (Custom basis choices will be supported in a later release, once ``qmmlpack-master`` is ready.)

Usage
=====

Once a fully-formed ``yaml`` config exists, ``autotune`` can be invoked with the command ``run_autotune config.yml`` from the shell, or alternatively through the ``cmlkit.autotune.core.run_autotune`` python method. Autotune will assume that it is running in a folder intended for this purpose, and create subfolders for logs, the resulting models, and caches. Currently, computed single (i.e. the different k-body terms are saved separately) MBTRs are automatically saved to disk so they don't have to be recomputed if only other parameters have changed.

Workflow for single-core run
----------------------------

* Convert the dataset you're working on into ``cmlkit`` format and place it in a location specified by the ``$CML_DATASET_PATH`` environment variable (see :doc:`dataset` and :doc:`globals`, as well as `the examples <https://github.com/sirmarcel/cmlkit-examples>`_)
* Verify that ``$CML_DATASET_PATH`` is set
* Create a directory for the project you're working on
* Write a config file (in that folder), omitting the ``parallel`` entry in the ``config`` block (or setting it to ``false``)
* While in that folder, run ``run_autotune config.yml`` (if you're logged into a remote computer via ssh, use the ``nh_run_autotune`` script, which will start the process in the background and makes sure it doesn't get terminated when the shell exits)
* You can observe the run in ``logs/name_given_in_config.log``
* Once it's finished, the best n models are saved to ``out/``
  
Post-processing of results is currently under active development, please check back soon.
  
Workflow for a parallel run
---------------------------

The steps are essentially identical, with the following modifications:

* Make sure ``parallel`` is set to ``true``
* A MongoDB instance has to be running:

    - Before starting autotune, navigate to the project directory and run ``prep_db config.yml``
    - This will generate a config file for MongoDB, which you can then start with ``mongod --config mongod_config.yml``
    - If you're on a server, the ``nh_start_db mongod_config.yml`` command can be used instead of starting ``mongod`` directly
      
* Worker processes need to be running, which is most easily achieved by using the ``start_worker -n N config.yml`` command, where you replace N with some short number or string identifying the worker you want to start. If you're logged in via ssh, use the ``nh_start_worker config.yml N`` script instead.
* Logs for the workers can be found in ``logs/worker_N.log``.
  
Note: For some reason, worker processes are resistant to termination with ``SIGTERM``, you have to send ``SIGKILL`` instead.

Note: You need to run ``prep_db`` in the project folder, and you need to re-run it every time the folder moves. I am aware this is not elegant. ;)