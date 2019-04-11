import time
from copy import deepcopy
import numpy as np
from collections import OrderedDict
import cmlkit2 as cml2
from cmlkit2.engine import BaseComponent


class SearchBase(BaseComponent):
    """Template class for Search"""

    def __init__(
        self, space, loss="rmse", maxevals=25, count_errors=True, seed=1, context={}
    ):
        super().__init__(context=context)
        self.loss = loss
        self.maxevals = maxevals
        self.count_errors = count_errors  # do failed evaluations count towards maxevals?
        self.seed = (
            seed
        )  # make sure to always use this seed; all searches should be deterministic.
        self.space = deepcopy(space)
        # your Search probably has to do
        # some additional parsing, do that in your __init__

        # every search is expected to meaningfully update
        # this as the search goes on:
        self._done = (
            False
        )  # True if optimisation is done (count > maxeval is done automatically)

        # this is taken care of by the submit/suggest methods
        self.suggestions = OrderedDict()
        self.losses = OrderedDict()

        self.count = 0  # number of evals (overall)
        self.count_ok = 0  # number of evals (ok)
        self.count_error = 0  # number of evals (error)
        self.best_count = 0  # count of best loss
        self.best_loss = float("inf")  # best loss seen so far
        self.best_tid = None  # tid of best loss seen so far

    def _suggest(self):
        raise NotImplementedError("Searches must implement a _suggest method.")
        # this function must return a dict-like object to be consumed by
        # an evaluator, and an id which you can use to internally
        # track evaluations (in case suggestions depend on previous results)

        # return (None, None) if no new evaluations are currently
        # desirable; for instance if the search space has been
        # exhausted. This will tell Runners to wait a little,
        # and hopefully stop running if you also set _done = True.
        return tid, suggestion

    def _submit(self, tid, loss, error=None, var=None):
        # for searches that don't use previous results,
        # we don't need to do anything with results!
        pass

    # you should not have to implement anything below in your subclass

    def _generate_tid(self):
        return cml2.engine.compute_hash(time.time() + np.random.rand())

    def suggest(self):
        tid, suggestion = self._suggest()
        self.suggestions[tid] = suggestion
        return tid, suggestion

    def submit(self, tid, result):
        # expected form of result at least: {self.loss: 123, 'status': 'ok'/'error'}
        # note that this assumes that
        # failed evaluations or malformed results
        # have infinite loss but are ok
        status = result.get("status", "ok")
        loss = result.get(self.loss, float("inf"))
        loss_variance = result.get(self.loss + "_var", None)

        if status == "error":
            error = result.get("error", (Exception.__class__.__name__, str(Exception)))
            self.count_error += 1
            cml2.logger.info(f"Received error in trial: {error}")
        else:
            error = None
            self.count_ok += 1

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_tid = tid
            self.best_count = self.count

        self.count += 1

        self.losses[tid] = loss

        self._submit(tid, loss, error=error, var=loss_variance)

    @property
    def done(self):
        if self.count_errors:
            return self._done or self.count >= self.maxevals
        else:
            return self._done or self.count_ok >= self.maxevals

    @property
    def since_last_improvement(self):
        return self.count - self.best_count

    @property
    def best_suggestion(self):
        if len(self.losses) == 0 or self.best_tid is None:
            return {}
        else:
            return self.suggestions[self.best_tid]

    @property
    def tids_by_loss(self):
        # sort self.losses by loss, so in the end
        # we have an ordered dict with tid -> loss mappings
        # ordered by increasing loss, and then makes a list out of
        # it, just keeping the tids
        # TODO: make this less insane
        return list(OrderedDict(sorted(self.losses.items(), key=lambda x: x[1])))
