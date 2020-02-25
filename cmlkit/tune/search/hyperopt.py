"""Hyperopt search algorithm."""

import numpy as np
from copy import deepcopy
import hyperopt as hpo
import logging
from itertools import count

from cmlkit.engine import Component
from cmlkit.utility.config_helpers import (
    find_pattern_apply_f,
    find_pattern,
    tuples_to_lists,
)


class Hyperopt(Component):
    """Interface to the hyperopt optimisers.

    The input space is expected to be a nested dict, where
    variables to be optimised are marked by

        ["hp_function", "variable_name", *args]

    Function can either be one of the functions natively supported by
    hyperopt (https://github.com/hyperopt/hyperopt/wiki/FMin), or one of
    two custom extensions:
        hp_loggrid(name, start, stop, num, base): returns hp.choice called on
            a logspace grid from start to stop with num steps (start and end are included)
        hp_loggrid_uniform(name, start, stop, q): returns hp.quniform, with its outputs
            then taken as `2**x`. While loggrid implicitly assumes no correlation between
            choices, this should (in theory) encode the assumption that inputs are related
            in log space. This is an experimental feature.

    This will also relucantly work with already-instantiated hyperopt functions,
    but this is a non-supported use and there is no guarantee things won't break.

    Parameters:
        space: dict with search space, see above.
        seed: seed for RNG.
        method: either "tpe" or "rand".
        errors_ok: if True, errors are treated as successful trials with infinite loss,
            otherwise they are ignored. (This is an experimental feature.)

    """

    kind = "search_hyperopt"

    def __init__(self, space, seed=123, method="tpe", errors_ok=False, context={}):
        super().__init__(context=context)

        self.og_space = deepcopy(space)

        self.space = deepcopy(space)
        # Parse space to contain hyperopt functions
        find_pattern_apply_f(self.space, is_hyperopt, to_hyperopt)
        self._needs_postprocessing = len(find_pattern(self.space, is_exp2)) > 0

        self.seed = seed

        self.method = method
        if method == "tpe":
            self.algo = hpo.tpe
        elif method == "rand":
            self.algo = hpo.rand
        else:
            raise ValueError(f"Do not recognise suggestion method {method}.")

        # determines whether errors are treated as errors, or as trials with infinite loss
        self.errors_ok = errors_ok

        # this is to avoid getting spammed with useless TPE diagnostic info
        algo_logger = logging.getLogger("hpo")
        algo_logger.setLevel(logging.WARNING)
        self.algo.logger = algo_logger

        # this is hyperopt internals
        self.domain = hpo.Domain(lambda spc: spc, self.space)
        self._hpopt_trials = hpo.Trials()

        # this maps internal hyperopt ids to trial ids
        self._live_trial_mapping = {}

        self.rstate = np.random.RandomState(self.seed)

        self.counter = count()  # used to generate unique, but deterministic trial ids

    def _get_config(self):
        return {
            "space": self.og_space,
            "seed": self.seed,
            "method": self.method,
            "errors_ok": self.errors_ok,
        }

    def suggest(self):
        tid = next(self.counter)

        suggestion = self._make_suggestion(tid)

        if self._needs_postprocessing:
            find_pattern_apply_f(suggestion, is_exp2, apply_exp2)

        return tid, suggestion

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
        # draws from the distributions. magic!
        config = hpo.base.spec_from_misc(new_trial["misc"])
        ctrl = hpo.base.Ctrl(self._hpopt_trials, current_trial=new_trial)
        memo = self.domain.memo_from_config(config)
        hpo.utils.use_obj_for_literal_in_memo(self.domain.expr, ctrl, hpo.base.Ctrl, memo)

        suggested_config = hpo.pyll.rec_eval(
            self.domain.expr,
            memo=memo,
            print_node_on_error=self.domain.rec_eval_print_node_on_error,
        )

        suggested_config = deepcopy(suggested_config)

        # hyperopt internally converts tuples to lists
        # but we treat *everything* as list in cmlkit
        # (background: yaml doesn't distinguish them,
        # and everything has to be able to go through yaml)
        tuples_to_lists(suggested_config)

        return suggested_config

    def submit(self, tid, error=False, loss=None, var=None):
        # Inform hyperopt about the results

        ho_trial = self._get_hyperopt_trial(tid)

        ho_trial["refresh_time"] = hpo.utils.coarse_utcnow()

        if error:
            if self.errors_ok:
                ho_trial["state"] = hpo.base.JOB_STATE_DONE
                ho_trial["result"] = self._to_hyperopt_result(float("inf"), None)
            else:
                ho_trial["state"] = hpo.base.JOB_STATE_ERROR

        else:
            assert loss is not None, "Must submit a loss as result if there was no error."
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


# at some future point, this will be transitioned to standard config syntax


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


def is_exp2(x):
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], str):
            return x[0] == "internal_exp2"


def apply_exp2(x):
    return 2 ** x[1]


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
        elif s[1] == "loggrid_uniform":
            return ["internal_exp2", hpo.hp.quniform(*x[1:])]
        else:
            raise NotImplementedError(
                "Hyperopt can't find function named {}!".format(s[1])
            )

    for a in args:
        # this takes care of nesting
        find_pattern_apply_f(a, is_hyperopt, to_hyperopt)

    f = f(*args)
    return f


def make_grid(label, start, stop, num, base=2.0):

    # it's very important to convert to a plain list here; otherwise
    # we carry numpy floats through the computation, which don't hash
    # to the same values as their plain float counterparts, which in
    # turn makes evaluation ids not match after a roundtrip through yaml.
    choices = np.logspace(start, stop, num=num, base=base).tolist()

    return (label, choices)
