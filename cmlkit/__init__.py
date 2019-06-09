import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

default_context = {}

from .engine import save_yaml, read_yaml, _from_config, _from_npy, _from_yaml

global classes
classes = {}


def from_config(config, context={}):
    return _from_config(config, classes=classes, context=context)


def from_npy(config, context={}):
    return _from_npy(config, classes=classes, context=context)


def from_yaml(config, context={}):
    return _from_yaml(config, classes=classes, context=context)


def register(*components):
    """Register Components with cmlkit for deserialisation."""
    for c in components:
        classes[c.kind] = c


from .env import (
    cache_location,
    dataset_path,
    get_scratch,
    runner_path,
    quippy_pythonpath,
    quippy_python_exe,
    get_plugins
)
from .utility import convert, unconvert, charges_to_elements, OptimizerLGS
register(OptimizerLGS)

from .dataset import Dataset, Subset, load_dataset
register(Dataset, Subset)

from .tune import components as components_tune
register(*components_tune)

from .evaluation import components as components_evaluation
register(*components_evaluation)

from .representation import components as components_representation
register(*components_representation)

from .regression import components as components_regression
register(*components_regression)

from .model import Model
register(Model)

from importlib import import_module

for plugin in get_plugins():
    p = import_module(plugin)
    components = getattr(p, "components", [])
    register(*components)
