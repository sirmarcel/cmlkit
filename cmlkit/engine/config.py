"""The infrastructure for deserialisation.

This provides the base functions that take care
of turning config dictionaries into actual objects.

Some global registry of classes has to be maintained
in order for this to work -- at the moment, this is
done in a global variable defined in the global `__init__.py`,
but in the future a more mature plugin system might be needed.

"""


from cmlkit.engine import BaseComponent, read_npy, read_yaml


def _from_npy(path, classes={}, **kwargs):
    config = read_npy(path)
    return _from_config(config, classes=classes, **kwargs)


def _from_yaml(path, classes={}, **kwargs):
    config = read_yaml(path)
    return _from_config(config, classes=classes, **kwargs)


def _from_config(config, classes={}, **kwargs):
    if isinstance(config, BaseComponent):
        # did we accidentally pass an already loaded thing?
        return config
    elif isinstance(config, dict):
        if "kind" not in config or "config" not in config:
            raise ValueError("Improper config format: " + str(config))

        class_name = config["kind"]
        if class_name in classes:
            return classes[class_name].from_config(config["config"], **kwargs)

        else:
            raise ValueError("Unknown class with name {}.".format(class_name))
    else:
        # TODO: attempt to load?
        raise ValueError("Config must be a dict type.")
