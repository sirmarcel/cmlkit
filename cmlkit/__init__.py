import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

default_context = {'cache_type': 'mem', 'min_duration': 0.5}

from .env import cache_location, dataset_path, scratch_location, runner_path, quippy_pythonpath, quippy_python_exe

from .core import get_loss, losses, LocalGridSearch, charges_to_elements
from .representations import MBTR1, MBTR2, MBTR3, MBTR4, ComposedRepresentation, OnlyCoords, OnlyDists, OnlyDistsHistogram, CoulombMatrixMinimal, BasicSymmetryFunctions, EmpiricalSymmetryFunctions, Soap
from .regressors import KRR, ExtensiveKRR
from .dataset import Dataset, Subset, View
from .model import Model

import engine
save_yaml = engine.save_yaml
read_yaml = engine.read_yaml

from conversion import convert, unconvert

from .tune import RunnerSingle, RunnerPool, SearchHyperopt, SearchFixed, Evals
from .evaluators import EvaluatorHoldout, EvaluatorLGS, EvaluatorCV

classes = {
    'mbtr1': MBTR1,
    'mbtr2': MBTR2,
    'mbtr3': MBTR3,
    'mbtr4': MBTR4,
    'composed': ComposedRepresentation,
    'only_coords': OnlyCoords,
    'only_dists': OnlyDists,
    'only_dists_histogram': OnlyDistsHistogram,
    'cm_minimal': CoulombMatrixMinimal,
    'bsf': BasicSymmetryFunctions,
    'esf': EmpiricalSymmetryFunctions,
    'krr': KRR,
    'ekrr': ExtensiveKRR,
    'soap': Soap,
    'model': Model,
    'dataset': Dataset,
    'subset': Subset,
    'evals': Evals,
    RunnerSingle.kind: RunnerSingle,
    RunnerPool.kind: RunnerPool,
    SearchHyperopt.kind: SearchHyperopt,
    SearchFixed.kind: SearchFixed,
    LocalGridSearch.kind: LocalGridSearch,
    EvaluatorHoldout.kind: EvaluatorHoldout,
    EvaluatorLGS.kind: EvaluatorLGS,
    EvaluatorCV.kind: EvaluatorCV
}


def from_config(config, context={}):
    return engine.from_config(config, classes=classes, context=context)


def from_npy(config, context={}):
    return engine.from_npy(config, classes=classes, context=context)


def from_yaml(path, context={}):
    config = read_yaml(path)
    return from_config(config, context=context)

from dataset_loader import load_dataset
