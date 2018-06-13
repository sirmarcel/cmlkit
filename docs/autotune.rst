********
Autotune
********

``cmlkit.autotune`` is the sub-module in charge of automated hyperparameter tuning for MBTR+KRR models. It's built on ``hyperopt`` (see the `repo <https://github.com/hyperopt/hyperopt>`_), which uses Tree-Structured Parzen Estimators to probabilistically optimise over the awkward search spaces arising when testing parameter combinations. ``hyperopt``, and consequently ``autotune``, also supports *parallelisation* with Mongo DB. In the following, I will give a brief overview on how to use ``autotune`` to optimise a model.

Overview
========

In essence, the input for ``hyperopt`` is a (nested, tree-shaped) dictionary of parameters, where the parameters to be optimised are replaced by probability distributions, representing an (intuitive) prior on their values. Crucially, this tree can contain nested expressions, so for instance we can express the fact that choosing one particular geometry function in the MBTR also requires us to choose a broadening, which is entirely unrelated to the one chosen for an unrelated geometry function. 

In technical terms, the priors are expressed by special functions, for instance ``hp.choice('name_of_param', ['option_1', 'option_2'])``, representing a uniformly random choice between two options. Continuous functions also exist, but are not usually used here -- caches are much easier to fill if only discrete choices are available. A list can be found in section 2.1 of `this wiki page <https://github.com/hyperopt/hyperopt/wiki/FMin>`_. 

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
    config:
      n_calls: 100
      cv: [random, 3, 1500, 500]  # cross-validation config; see below
      # loss: rmse  # string specifying the loss to optimise, taken directly from cmlkit.losses
      # n_cands: 2  # number of candidate models to save (ranked by loss)
      # loglevel: INFO  # amount of logging, DEBUG prints a lot of internal stuff
      # timeout: None # if set, number of seconds after which a trial is aborted
      # parallel: false # if true, parallel mode with MongoDB will be engaged
      # if parallel is true, the following defaults are also applied:
      # db_name: project_name
      # db_port: 1234
      # db_ip: 127.0.0.1
    spec:
      # See below

If a config file is passed to autotune, the following transformations are applied:

* Losses are replaced with the appropriate functions (see :doc:`losses` for a list)
* Grid specifications are replaced with the actual grids (see below)
* Hyperopt specifications are replaced with hyperopt functions (see below)

Cross Validation
----------------
    
Currently, only random cross-validation is supported, with the following syntax: ``[random, n_cv, n_train, n_test]``. This means that ``n_cv`` times, the dataset is *randomly* split into ``n_train`` structures to train and ``n_test`` structures to predict. The loss is then computed for each, and averaged over the repetitions.

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

In principle, all ``hyperopt`` functions are supported, but in practice, we currently only use ``hp_choice`` (it might be worth investigating this in more detail). For ``hp_choice``, the first argument has to be a string uniquely identifying the parameter to be chosen. The second argument has to be a list of possible options. In the case of numerical parameters, it is tedious to specify these lists by hand, and so the ``grid`` module exists...

Note that, the choices can be *nested* and *conditional*. For instance, in the above, some choices of ``weightf`` require an additional parameter, which is also chosen with ``hp_choice``. The same principle applies when whole MBTR configurations are put into ``hp_choice``. In such cases, some care has to be taken to keep the labels of parameters unique.

Grid Specifications
-------------------

Prefixed with ``gr_``. A few convenience shortcuts have been implemented, for instance ``gr_medium`` is a grid evenly spaced in log2 space from -18 to +20 in steps of 1. These shortcuts currently require the use of a one-item list, which is somewhat awkward, this will be changed at some point. 

``gr_log2`` generates a log2 grid. A full list can be found in :doc:`grids`.
