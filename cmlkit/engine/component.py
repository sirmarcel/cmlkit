"""Base class.

The main objects in cmlkit follow a pattern similar to Keras models:

They can be serialised into and instantiated from `config` dictionaries.
Their syntax is specified in the `engine.config` module.

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
from .config import Configurable


class Component(Configurable):
    """Base class for cmlkit components.

    Provides the mechanism for passing context (see explanation above)
    to objects.

    All context variables must have a default value set in the `default_context`
    of the class. This is to ensure that passing an empty context will never
    result in an error.

    """

    # define this in subclass if certain context variables are required
    default_context = {}

    def __init__(self, context={}):  # pylint: disable=dangerous-default-value
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
