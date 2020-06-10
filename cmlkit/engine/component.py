"""Component."""

from cmlkit import logger
from .config import Configurable
from .hashing import compute_hash
from .cache import Cached


class Component(Configurable, metaclass=Cached):
    """Base class for cmlkit components.

    See explanation in readme!
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

    def register_cache(self):
        from cmlkit import caches

        self.cache = caches.register(self)

    def get_hash(self):
        """Hash of this component"""
        return self.get_config_hash()

    def get_hid(self):
        """History/human readable ID (kind@hash)

        This id is used in history tracking of Data
        instances, the idea is to be *slighlty* less
        opaque than just a hash.
        """
        return f"{self.get_kind()}@{self.get_hash()}"
