"""Parsing of config dictionaries.

The config format is:

```
    {
    "kind": { # inner config }
    }
```

"kind" is the name of the class to be instantiated.
"inner config" is a an iterable specifying how this class is created.
"config" is the combination of both.

We use the term "kind" rather than class for two reasons
- `class` is a keyword in python, so it can't be used in code
- this config format is also used for functions

"""

from collections.abc import Iterable


def is_config(config):
    if isinstance(config, dict):
        if len(config) == 1:
            kind = next(iter(config))
            if isinstance(kind, str):
                if isinstance(config[kind], Iterable):
                    return True

    return False


def parse_config(config, shortcut_ok=False):
    if shortcut_ok and isinstance(config, str):
        return config, {}

    if not is_config(config):
        raise ValueError("Improper config format: " + str(config))

    kind = next(iter(config))
    inner = config[kind]

    return kind, inner
