"""Interface to qmmlpack local grid search optimiser."""

import qmmlpack
import time

from cmlkit import logger
from cmlkit.engine import Component


class OptimizerLGS(Component):
    """Wraps the qmmlpack.local_grid_search method."""

    kind = "opt_lgs"

    def __init__(self, resolution=None, rng=None, maxevals=None, context={}):
        super().__init__(context=context)

        self.resolution = resolution
        self.rng = rng
        self.maxevals = maxevals

    def _get_config(self):
        return {"resolution": self.resolution, "rng": self.rng, "maxevals": self.maxevals}

    def __call__(self, f, variables):
        start = time.monotonic()

        logger.info(
            "Starting local grid search with {} variables.".format(len(variables))
        )

        result = qmmlpack.local_grid_search(
            f=f,
            variables=variables,
            evalmonitor=log_during_eval,
            resolution=self.resolution,
            maxevals=self.maxevals,
            rng=self.rng,
        )

        end = time.monotonic()
        result["duration"] = end - start
        result["best"] = result["best_valpow"]

        logger.info(
            f"Ended local grid search with loss {result['best_f']} at {result['best']}."
        )

        return result


def log_during_eval(trialv, trialf, bestv, bestf, state):
    logger.debug(
        "Step {}/{}, f={} ({})".format(
            state["num_evals"], state["max_evals"], trialf, bestf
        )
    )
