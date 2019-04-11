from cmlkit import logger
from cmlkit.engine.inout import save_yaml


class Configurable():
    """Mixing class for serialising/de-serialising objects as dictionaries"""


    @classmethod
    def from_config(cls, config, **kwargs):
        """Instantiate this class from config

        # TODO proper docstring

        """
        if 'config' in config and 'kind' in config:
            # we have been handed a fully formed config, not one for this class
            return cls.from_config(config['config'], **kwargs)
        else:
            return cls._from_config(config, **kwargs)

    @classmethod
    def _from_config(cls, config, **kwargs):
        # TODO: figure out how to avoid passing useless keyword args along
        # currently every _from_config must accept pointless keyword arguments
        # this method must be implemented
        # and instantiate this object
        raise NotImplementedError('Configurables must implement a _from_config method.')

    def get_config(self):
        """Return a dictionary describing this component"""
        return {'kind': self.get_kind(), 'config': self._get_config()}

    def get_kind(self):
        # 'kind' is used to identify components
        # for de-serialisation; by default this will
        # be the lowercase version of the class name,
        # or manually defined as class-level attribute 'kind'
        return getattr(self.__class__, 'kind', self.__class__.__name__.lower())

    def _get_config(self):
        # this method must return a dict that fully
        # described how to re-instantiate this component
        raise NotImplementedError('Configurables must implement a get_config method')

    def save_config(self, filename=None):
        config = self.get_config()

        if filename is None:
            if hasattr(self, 'name'):
                name = self.name
            else:
                name = self.get_kind()

        save_yaml(filename, config)




class BaseComponent(Configurable):
    """Base class for cmlkit model components"""

    # define this in subclass if certain context variables are required
    default_context = {}

    def __init__(self, context={}):
        # import global context here so that it can be changed dynamically in a running session
        from cmlkit import default_context as global_default_context

        # special case: if context includes a sub-dict
        # under the key that is the name of this class,
        # use this context instead (important for nested things)
        if self.get_kind() in context:
            # the context for this class is expanded into this
            # class' context, while retaining the full dict,
            # so if this component instantiates other components,
            # they get a shot at having some context passed to them
            # as well!
            context = {**context, **context[self.get_kind()]}

        self.context = {**global_default_context, **self.__class__.default_context, **context}

        logger.debug(f"Context for {self.get_kind()} is {self.context}.")

        # you should probably also implement something more here
