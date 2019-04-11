from cmlkit.engine import BaseComponent, read_npy


def _from_npy(path, context={}, classes={}):
    config = read_npy(path)
    return from_config(config, context=context, classes=classes)


def _from_config(config, classes={}, context={}):
    if isinstance(config, BaseComponent):
        # did we accidentally pass an already loaded thing?
        return config
    elif isinstance(config, dict):
        if 'kind' not in config or 'config' not in config:
            raise ValueError('Improper config format: ' + str(config))

        class_name = config['kind']
        if class_name in classes:
            class_to_instantiate = classes[class_name]

            if hasattr(class_to_instantiate, 'from_config'):
                return class_to_instantiate.from_config(config['config'], context=context)
            else:
                return class_to_instantiate(**config['config'], context=context)

        else:
            raise ValueError('Unknown class with name {}.'.format(class_name))
    else:
        # TODO: attempt to load?
        raise ValueError('Config must be a dict type.')


def to_config(thing):
    return thing.to_config()
