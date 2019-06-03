"""Infrastructure to tune KRR parameters."""

import qmmlpack
from functools import partial
import numpy as np

from cmlkit.engine import Component
from cmlkit.evaluation import get_lossf
from cmlkit import from_config


class KRRTunerCV(Component):
    """KRR intended only for parameter tuning with CV."""

    kind = "krr_tuner_cv"

    def __init__(
        self,
        kind_kernel,
        kind_kernelf,
        optimizer,
        centering=False,
        lossf="rmse",
        context={},
    ):
        super().__init__(context=context)

        self.optimizer = from_config(optimizer, context=context)
        self.lossf = get_lossf(lossf)

        self.kind_kernel = kind_kernel
        self.kind_kernelf = kind_kernelf

        self.centering = False

    def prepare(self, x, y, idx):

        n_splits = len(idx)

        def kernel(ls):
            actual_kernel = _make_kernel(ls, self.kind_kernel, self.kind_kernelf)
            return actual_kernel(x)

        # kernel = memcached(kernel)

        def target(nl, ls):

            kernel_matrix = kernel(ls)

            loss = 0.0
            for train, test in idx:
                kernel_train = kernel_matrix[np.ix_(train, train)]
                kernel_test = kernel_matrix[np.ix_(train, test)]

                krr = qmmlpack.KernelRidgeRegression(
                    kernel_train, y[train], theta=(nl,), centering=self.centering
                )

                pred = krr(kernel_test)
                true = y[test]

                loss += self.lossf(true, pred) / n_splits

            return loss

        self.ready = True
        self.target = target

    def tune(
        self, args_nl=(0, 1, 0.5, -20, 2, -1, 2.0), args_ls=(0, 2, 0.5, -15, 15, +1, 2.0)
    ):
        assert self.ready, "KRRTunerCV not ready for action."

        result = self.optimizer(self.target, (args_nl, args_ls))

        config_kernel = _make_kernel(
            result["best"][1], self.kind_kernel, self.kind_kernelf
        ).get_config()
        config_krr = {"krr": {"nl": result["best"][0], "kernel": config_kernel}}

        return config_krr, result


def _make_kernel(ls, kind_kernel, kind_kernelf):
    config = {kind_kernel: {"kernelf": {kind_kernelf: {"ls": ls}}}}

    return from_config(config)
