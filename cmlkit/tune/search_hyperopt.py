import numpy as np
import copy
import time
import logging
import hyperopt as hpo

from cmlkit import logger
from ..engine import compute_hash
from .search_base import SearchBase
from ..helpers import find_pattern_apply_f


class SearchHyperopt(SearchBase):
    """A wrapper around HyperOpt to provide trial suggestions.

    Adapted from https://github.com/ray-project/ray/tree/master/python/ray/tune,
    a part of ray, which is licensed under Apache 2.0 as stated here:
    https://github.com/ray-project/ray/blob/master/LICENSE

    Expects the space to be formatted as something like
    {
          'nl': ['hp_qloguniform', 'nl', -10, 2, 0.1],
          'kernel': ['gaussian', ['hp_qloguniform', 'ls', -2, 10, 0.1]]
    }

    Where after the _ any named hyperopt function (see https://github.com/hyperopt/hyperopt/wiki/FMin)
    can be put. It is recommended to stick to discrete versions, in order to allow caching to work well.
    """

    kind = "search_hyperopt"

    def __init__(
        self,
        space,
        loss="rmse",
        max_concurrent=10,
        maxevals=1000,
        count_errors=True,
        seed=123,
        method="tpe",
        context={},
    ):

        super().__init__(
            space,
            loss=loss,
            maxevals=maxevals,
            count_errors=count_errors,
            seed=seed,
            context=context,
        )

        self.parsed_space = copy.deepcopy(self.space)

        # Parse space to contain hyperopt functions
        find_pattern_apply_f(self.parsed_space, is_hyperopt, to_hyperopt)

        # The number of suggestions that can be requested before
        # the search demands some results first: It is not helpful to
        # set this too large, because then we waste computations on
        # trials that have not taken advantage of previous trials, i.e.
        # the search becomes basically random. This should probably be set
        # to around the same (or slightly larger) than the number of workers
        # if running in parallel.
        self.max_concurrent = max_concurrent

        self.method = method
        if method == "tpe":
            self.algo = hpo.tpe
        elif method == "rand":
            self.algo = hpo.rand
        else:
            raise ValueError(f"Do not recognise suggestion method {method}.")

        # this is to avoid getting spammed with useless TPE diagnostic info
        algo_logger = logging.getLogger("hpo")
        algo_logger.setLevel(logging.WARNING)
        self.algo.logger = algo_logger

        # this is hyperopt internals
        self.domain = hpo.Domain(lambda spc: spc, self.parsed_space)
        self._hpopt_trials = hpo.Trials()

        # this maps internal hyperopt ids to trial ids
        self._live_trial_mapping = {}

        self.rstate = np.random.RandomState(self.seed)

    @classmethod
    def _from_config(cls, config, context={}):
        defaults = {
            "loss": "rmse",
            "max_concurrent": 10,
            "maxevals": 1000,
            "seed": 123,
            "method": "tpe",
            "count_errors": True,
        }
        return cls(**{**defaults, **config}, context=context)

    def _get_config(self):
        return {
            "space": self.space,
            "loss": self.loss,
            "max_concurrent": self.max_concurrent,
            "maxevals": self.maxevals,
            "count_errors": self.count_errors,
            "seed": self.seed,
            "method": self.method,
        }

    def _make_suggestion(self, tid):
        # this is where we call hyperopt

        # tell hyperopt that we want one new trial
        new_ids = self._hpopt_trials.new_trial_ids(1)
        self._hpopt_trials.refresh()

        # Get new suggestion from Hyperopt;
        new_trials = self.algo.suggest(
            new_ids, self.domain, self._hpopt_trials, self.rstate.randint(2 ** 31 - 1)
        )

        self._hpopt_trials.insert_trial_docs(new_trials)
        self._hpopt_trials.refresh()
        new_trial = new_trials[0]

        # keep track of this trial with our tid
        self._live_trial_mapping[tid] = (new_trial["tid"], new_trial)

        # Taken from HyperOpt.base.evaluate; I think this one actually
        # draws from the distributions
        config = hpo.base.spec_from_misc(new_trial["misc"])
        ctrl = hpo.base.Ctrl(self._hpopt_trials, current_trial=new_trial)
        memo = self.domain.memo_from_config(config)
        hpo.utils.use_obj_for_literal_in_memo(self.domain.expr, ctrl, hpo.base.Ctrl, memo)

        suggested_config = hpo.pyll.rec_eval(
            self.domain.expr,
            memo=memo,
            print_node_on_error=self.domain.rec_eval_print_node_on_error,
        )

        return copy.deepcopy(suggested_config)

    def _suggest(self):
        if self._num_live_trials() >= self.max_concurrent or self.done:
            logger.debug(
                "Hyperopt has reached max_concurrent trials, please submit some results before asking for more suggestions."
            )
            return None, None

        tid = self._generate_tid()

        suggestion = self._make_suggestion(tid)

        return tid, suggestion

    def _submit(self, tid, loss, error=None, var=None):
        # Inform hyperopt about the results

        ho_trial = self._get_hyperopt_trial(tid)

        # I don't think this case should happen; but it
        # does...
        # TODO: find out when this occurs.
        if ho_trial is None:
            return None

        ho_trial["refresh_time"] = hpo.utils.coarse_utcnow()

        if error is not None:
            ho_trial["state"] = hpo.base.JOB_STATE_ERROR
            ho_trial["misc"]["error"] = error

        else:
            ho_trial["state"] = hpo.base.JOB_STATE_DONE
            ho_trial["result"] = self._to_hyperopt_result(loss, var)

        self._hpopt_trials.refresh()
        del self._live_trial_mapping[tid]

    def _to_hyperopt_result(self, loss, var):
        if var is None:
            return {"loss": loss, "status": "ok"}
        else:
            return {"loss": loss, "status": "ok", "loss_variance": var}

    def _get_hyperopt_trial(self, trial_id):
        if trial_id not in self._live_trial_mapping:
            return
        hyperopt_tid = self._live_trial_mapping[trial_id][0]
        return [t for t in self._hpopt_trials.trials if t["tid"] == hyperopt_tid][0]

    def _num_live_trials(self):
        return len(self._live_trial_mapping)


def is_hyperopt(x):
    """Check whether a given object is a hyperopt argument

    The format expected is ('hp_NAME_OF_FUNCTION', 'name for hyperopt', remaining, arguments)

    """

    if isinstance(x, (tuple, list)):
        if isinstance(x[0], str):
            s = x[0].split("_", 1)
            if s[0] == "hp":
                return True

    return False


def to_hyperopt(x):
    """Convert a sequence to a hyperopt function

    Example: ('hp_choice', 'mbtr_1', [1, 2, 3])
             -> hp.choice('mbtr_1', [1, 2, 3])

    """
    # print(f"Converting {x}")

    s = x[0].split("_", 1)

    try:
        f = getattr(hpo.hp, s[1])
        args = x[1:]
    except AttributeError:
        # implement a custom hp function: hp_loggrid,
        # which first generated a base2 loggrid and
        # then applies hp.choice to it.
        if s[1] == "loggrid":
            f = hpo.hp.choice
            args = make_grid(*x[1:])
        else:
            raise NotImplementedError(
                "Hyperopt can't find function named {}!".format(s[1])
            )

    # print(f"f={f}")
    # print(f"args={args}")
    for a in args:
        # this takes care of nesting
        find_pattern_apply_f(a, is_hyperopt, to_hyperopt)

    f = f(*args)
    # print(f"returning {f}")
    return f


def make_grid(label, start, stop, num, base=2.0):

    choices = np.logspace(start, stop, num=num, base=base)

    return (label, choices)
