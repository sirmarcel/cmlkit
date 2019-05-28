"""Wrapper for the pebble ProcessPool."""

from pebble import ProcessPool
from diskcache import Index
import traceback

from cmlkit import from_config
from cmlkit.engine import compute_hash


class EvaluationPool:
    """Wrapper around ProcessPool.

    Essentially a ProcessPool that can only evaluate configs,
    with some tweaks.

    Aims to do the following:
        - Instantiate evaluators once in the workers to reduce overhead
        - Use a cache (provided by `diskcache`) to not have to run tasks twice
        - Provide common format for results, with support for "ok" and "error" status
        - Catches specified, but not all, exceptions
        - Provides timeouts backed by a sufficiently brutal approach to killing processes*

    We therefore sacrifice a little bit of generality for convenience
    in our particular domain, which is just how we like it.

    Note that this deliberately doesn't do any event loop management, it simply provides
    the `.schedule` function which schedules the evaluation of a config, and a `.finish`
    function with obtains the result, with error checking, of that evaluation.

    ***

    WARNING: hash keys in the cache (called "evals") are solely based on
    the config to be evaluated. So if some `evals` with a different underlying evaluator
    get passed, things will break in undefined ways. Since you are not expected
    to ever touch this clas without `Run` mediating, I don't think this is a problem.

    But be careful out there.

    ---
    * We interface with external code that doesn't always play by the rules, and in
      particular is quite fond of not reacting to SIGTERM. The `concurrent.futures`
      `ProcessPoolExecutor` doesn't seem to be able to enforce a timeout in such cases.

    """

    def __init__(
        self,
        max_workers,
        evals,
        evaluator_config,
        evaluator_context={},
        trial_timeout=None,
        caught_exceptions=(),
    ):
        self.trial_timeout = trial_timeout
        self.pool = ProcessPool(
            initializer=initializer,
            initargs=(evaluator_config, evaluator_context, evals),
            max_workers=max_workers,
        )

        self.evals = evals  # database of evaluations, acting as cache
        self.caught_exceptions = caught_exceptions

    def schedule(self, config):
        """Schedule evaluation of a config."""

        eid = compute_hash(config)
        future = self.pool.schedule(
            evaluate, args=(eid, config), timeout=self.trial_timeout
        )

        future.eid = eid  # annotate with hash key in evals

        return future

    def finish(self, future):
        """Obtain result of an evaluation future, catch errors, update caches.

        Should only be called with a finished future... but it's not a problem
        if it's not. The call to `future.result()` will trigger execution."""

        try:
            result = future.result()
        except self.caught_exceptions as e:
            trace = traceback.format_exc()
            result = {
                "error": {
                    "error": e.__class__.__name__,
                    "error_text": str(e),
                    "traceback": trace,
                }
            }
            self.evals[future.eid] = result

        return result


def initializer(evaluator_config, evaluator_context, index_evals):
    global evaluator
    global evals
    evals = index_evals
    evaluator = from_config(evaluator_config, evaluator_context)


def evaluate(eid, config):

    if eid in evals:
        return evals[eid]
    else:
        eval_result = evaluator(config)
        result = {"ok": eval_result}

        evals[eid] = result
        return result
