"""Manage 'lazily importing' qmmlpack."""

import cmlkit


class ImporterQmmlpack:
    """Utility class to lazy-import qmmlpack, or throw an error."""

    def __init__(self):
        self.qmmlpack = None

    def __call__(self, task="use this functionality"):
        if self.qmmlpack is None:
            try:
                import qmmlpack as qmml
            except ImportError:
                raise cmlkit.DependencyMissing(
                    f"To {task} you need to install qmmlpack (development branch): https://gitlab.com/qmml/qmmlpack/-/tree/development"
                )
            self.qmmlpack = qmml

        return self.qmmlpack


class ImporterQmmlpackExperimental:
    """Utility class to lazy-import qmmlpack.experimental, or throw an error."""

    def __init__(self):
        self.qmmlpack = None

    def __call__(self, task="use this functionality"):
        if self.qmmlpack is None:
            try:
                import qmmlpack.experimental as qmml
            except ImportError:
                raise cmlkit.DependencyMissing(
                    f"To {task} you need to install qmmlpack (development branch): https://gitlab.com/qmml/qmmlpack/-/tree/development"
                )
            self.qmmlpack = qmml

        return self.qmmlpack


import_qmmlpack = ImporterQmmlpack()
import_qmmlpack_experimental = ImporterQmmlpackExperimental()
