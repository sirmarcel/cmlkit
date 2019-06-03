"""Regression methods.

Currently, only kernel ridge regression as implemented by
`qmmlpack` is supported. Once additional regressors are
supported, the interface will be abstracted. For now, please
check `qmml/krr.py` for the canonical regression method
interface.

"""

from .qmml import components as qmml_components

components = qmml_components
