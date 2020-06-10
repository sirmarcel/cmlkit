"""Regression methods."""

from .data import KernelMatrix
from .kernel import Kernel
from .qmml import components as qmml_components

components = [*qmml_components, KernelMatrix]
