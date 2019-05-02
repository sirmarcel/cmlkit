"""Base classes.

The main objects in cmlkit follow a pattern similar to Keras models:

They can be serialised into and instantiated from `config` dictionaries.

The underlying idea is that these objects are essentially convenience wrappers
around functions with lots of arguments (for instance a regression method or
a representation), and don't have a lot of state (ideall they'd have none), so
once instantiated, they simply do some computation on inputs.

The config dict then simply specifies what kind of computation is done.

Since sometimes it is necessary to specify some arguments that do not influence
the outcome of the computation, but determinse how it's executed (for instance
how many cores to use, or cache settings), cmlkit objects also have a `context`,
another dictionary that is passed to the constructor as optional keyword argument.

So when implementing custom classes, the rules are:

a) The config must fully determine the output,
b) Everything else goes in the context.

Why am I doing things this way? Here are some reasons:

- This gives a consistent way to share model architectures as plaintext (yaml)
- This makes hashing and caching a breeze (no trying to hash a complex object)
- It gives full control over serialisation (I don't understand pickling at all)

Also, in the future, this will make it relatively easy to provide a fully functional
interface that avoids side effects.

"""

from cmlkit import logger
from cmlkit.engine.inout import save_yaml


class Configurable:
    """Mixin for serialising/de-serialising objects as dictionaries.

    The goal is to provide a simple, robust, human-readable way to
    pass objects around that have little to no state and are mainly
    containers for a bunch of arguments/parameters.

    This representation is the *config*, a tree-structured
    nested dictionary. It's core structure is:

    ```
    {
    "kind": "class_name",
    "config": { ... },
    }
    ```

    "kind" will be used to look up the class to instantiate, and the
    "config" dict will be used to then instantiate. Typically, this will
    be some variation on calling `__init__(**config)`.

    The term "kind" is used because class is a keyword in Python.

    """

    @classmethod
    def from_config(cls, config, **kwargs):
        """Instantiate this class from config."""

        # Note: We retain the **kwargs for future flexibility,
        # at the moment it is only needed to pass along the context.

        if "config" in config and "kind" in config:
            # we have been handed a fully formed config, not one for this class
            return cls.from_config(config["config"], **kwargs)
        else:
            return cls._from_config(config, **kwargs)

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls(**config, **kwargs)

    def get_config(self):
        """Return a dictionary describing this component"""
        return {"kind": self.get_kind(), "config": self._get_config()}

    def get_kind(self):
        """Return the class identifier."""
        # 'kind' is used to identify components
        # for de-serialisation; by default this will
        # be the lowercase version of the class name,
        # or manually defined as class-level attribute 'kind'
        return getattr(self.__class__, "kind", self.__class__.__name__.lower())

    def _get_config(self):
        # this method must return a dict that fully
        # described how to re-instantiate this component
        raise NotImplementedError("Configurables must implement a get_config method")


class BaseComponent(Configurable):
    """Base class for cmlkit components.

    Provides the mechanism for passing context (see explanation above)
    to objects.

    All context variables must have a default value set in the `default_context`
    of the class. This is to ensure that passing an empty context will never
    result in an error.

    """

    # define this in subclass if certain context variables are required
    default_context = {}

    def __init__(self, context={}):
        # import global context here so that it can be changed dynamically in a running session
        from cmlkit import default_context as global_default_context

        # special case: if context includes a sub-dict
        # under the key that is the name of this class,
        # use this context instead (important for nested instantiation)
        if self.get_kind() in context:
            # the context for this class is expanded into this
            # class' context, while retaining the full dict,
            # so if this component instantiates other components,
            # they get a shot at having some context passed to them
            # as well!
            context = {**context, **context[self.get_kind()]}

        self.context = {
            **global_default_context,
            **self.__class__.default_context,
            **context,
        }

        logger.debug(f"Context for {self.get_kind()} is {self.context}.")

        # you should probably also implement something more here
