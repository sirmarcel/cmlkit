"""The infrastructure for de/serialisation.

See base_component for reasoning.

This module provides:
    - Functions to turn configs into objects
    - Mixin for making objects seriasable in this way

Some global registry of classes has to be maintained
in order for this to work -- at the moment, this is
done in a global variable defined in the global `__init__.py`,
but in the future a more mature plugin system might be needed.

"""


from cmlkit.engine import read_npy, read_yaml, is_config, parse_config
from cmlkit import logger


def _from_npy(path, classes={}, **kwargs):
    config = read_npy(path)
    return _from_config(config, classes=classes, **kwargs)


def _from_yaml(path, classes={}, **kwargs):
    config = read_yaml(path)
    return _from_config(config, classes=classes, **kwargs)


def _from_config(config, classes={}, **kwargs):
    if isinstance(config, Configurable):
        # did we accidentally pass an already instantiated object?
        return config
    else:
        kind, inner = parse_config(config)
        if kind in classes:
            return classes[kind].from_config(inner, **kwargs)
        else:
            raise ValueError(f"Cannot find class with name {kind} in registry.")


def to_config(configurable):
    """Turn into config, if possible."""
    if isinstance(configurable, Configurable):
        return configurable.get_config()
    elif is_config(configurable):
        return configurable
    else:
        raise ValueError(f"Can't turn {configurable} into config.")


class Configurable:
    """Mixin for serialising/de-serialising objects as dictionaries.

    The goal is to provide a simple, robust, human-readable way to
    pass objects around that have little to no state and are mainly
    containers for a bunch of arguments/parameters.

    This representation is the *config*, a tree-structured
    nested dictionary. It's core structure is:

    ```
    {
    "kind": { # inner config }
    }
    ```

    "kind" will be used to look up the class to instantiate, and the
    "inner config" dict will be used to then instantiate. Typically, this will
    be some variation on calling `__init__(**config)`.

    For more information see configparse.

    """

    @classmethod
    def from_config(cls, config, **kwargs):
        """Instantiate this class from (inner) config."""

        # Note: We retain the **kwargs for future flexibility,
        # at the moment it is only needed to pass along the context.

        # Keeping this proxy method allows us to add further functionality as needed,
        # while leaving sub-classes free to implement their own instantiation logic.

        return cls._from_config(config, **kwargs)

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls(**config, **kwargs)

    def get_config(self):
        """Return a dictionary describing this component"""
        return {self.get_kind(): self._get_config()}

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
