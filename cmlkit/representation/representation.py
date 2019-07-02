"""Define representation base class."""

import numpy as np

from cmlkit.engine import Component


class Representation(Component):
    """Representation base class.

    Representations are various ways of transforming the raw geometry
    of periodic or non-periodic systems into feature vectors, on which
    a regression method can then be trained.

    In `cmlkit`, representations are essentially functions that take
    a `Dataset` as input and produce a computed representation, either

    a) an array of length `n_systems` for global representations, or
    b) a list (length `n_systems`) of arrays (length `n_atoms`) with atomic
       representations.

    Representations typically have some parameters. In order to cleanly separate these
    parameters from the data on which a representation is constructed, the configuration
    of a representation happens upon instantiation,
    while the data is passed into the actual `__call__` method.

    Once instantiated, a representation therefore acts as a (mostly) pure function.
    Therefore, changing any attributes after instantiation is STRONGLY discouraged.

    This base class is used to define the interface for a `Representation`, and
    to provide a central place to implement caching and other common functionality.

    The recommended pattern for implementing a representation is as follows:

    - Decide on a canonical syntax for parameters, ideally following the `cmlkit`
      config pattern.
    - Implement a module-level function of the form compute_rep(data, config). Within
      this function, perform the translation of your syntax into the underlying syntax
      of the code that *actually* implements the representation.
    - In the constructor for the representation, transform the args into canonical form.
      (Avoid setting too many attributes -- that just encourages mistakes.)

    """

    def __init__(self, context={}):
        # can't use default_context because subclasses overwrite it
        context = {"chunk_size": None, **context}
        super().__init__(context=context)

    def __call__(self, data):
        """Compute this representation."""
        if self.context["chunk_size"] is None:
            return self.compute(data)
        else:
            chunks = [
                self.compute(chunk)
                for chunk in data.in_chunks(size=self.context["chunk_size"])
            ]
            return np.concatenate(chunks, axis=0)

    def compute(self, data):
        raise NotImplementedError("Representations must implement a compute method.")
