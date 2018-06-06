**********
Model Spec
**********

Imports
=======

Convenience import ``cmlkit.ModelSpec``, actually ``cmlkit.model_spec``.

Introduction
============

The ``ModelSpec`` class exists to collect all information necessary to build a prediction model with the MBTR and kernel ridge regression. (Currently -- in the future, alternative representations and regression methods might get implemented.) It is essentially a wrapper around a two dictionaries ``krr`` and ``mbtrs``, holding the specifications for each.

The ``ModelSpec`` class is designed to allow easy saving and loading from disk in a human-readable format: `YAML <https://yaml.org>`_. Therefore, you are most likely to create ``ModelSpec``s with the ``from_yaml`` method. Alternatively, a ``from_dict`` class method also exists. Creation via the constructor is also possible, but should usually be avoided.

Saving happens with the ``save`` method. The default extension is ``.spec.yml``.


YAML/Dictionary Syntax
======================

The following is a minimal template for a ``yaml`` representation of a ``ModelSpec``:

.. code-block:: yaml

    name: 'model_mini'
        desc: 'A minimal example model'
        data:  # what data is this model built for?
          id: 'dataset_id'
          property: 'fec'  # what property should it predict?
        mbtrs:
          mbtr_1:
            k: 1
            d: [-0.5, 1.6333333333333333, 30]
            geomf: 'count'
            distrf: ['normal', 11.313708499]
            weightf: '1/identity'
            eindexf: 'full'
            aindexf: 'full'
            # the following are optional, 
            # the given values the defaults
            corrf: 'identity'
            norm: null
            flatten: true
            elems: null
            acc: 0.001
            

        krr:
          kernelf: ['gaussian', 0.3535533905932738]
          nl: 6.103515625e-05  # the regularisation parameter
          centering: false

Note that flowstyle (as above) is preferred for the argument lists, but not for the dictionary portion.

The dictionary version is exactly identical (only substituting ``None`` for ``null``), all above keys (``name, desc, data, krr, mbtrs``) must be present. The ``mbtrs`` dictionary can contain arbitrarily many sub-dictionaries specifying individual k-body MBTRs. When the MBTR is computed, these will be concatenated. 

The arguments for the MBTR are identical to the ones in ``qmmlpack``. All kernels (``linear``, ``gaussian``, ``laplacian``) are supported. 

Docs
====


.. autoclass:: cmlkit.ModelSpec
    :members:
    :undoc-members:
