import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

default_context = {"cache_type": "mem", "min_duration": 0.5}

classes = {}

from .env import (
    cache_location,
    dataset_path,
    scratch_location,
    runner_path,
    quippy_pythonpath,
    quippy_python_exe,
)
from .conversion import convert, unconvert

from .core import get_loss, losses, LocalGridSearch, charges_to_elements

from .engine import save_yaml, read_yaml, _from_config, _from_npy, _from_yaml

from .dataset_loader import load_dataset

from .tune import classes as classes_tune
from .evaluators import classes as classes_evaluators
from .representations import classes as classes_representations
from .regression import classes as classes_regression
from .model import Model
from .dataset import Dataset, Subset

classes = {
    **classes_tune,
    **classes_evaluators,
    **classes_representations,
    **classes_regression,
    LocalGridSearch.kind: LocalGridSearch,
    Dataset.kind: Dataset,
    Subset.kind: Subset,
}

# 'model': Model,
# 'dataset': Dataset,
#     'subset': Subset,


def from_config(config, context={}):
    return _from_config(config, classes=classes, context=context)


def from_npy(config, context={}):
    return _from_npy(config, classes=classes, context=context)


def from_yaml(config, context={}):
    return _from_yaml(config, classes=classes, context=context)
