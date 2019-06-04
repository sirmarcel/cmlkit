"""Wrapper for the pebble ProcessPool."""

from pebble import ProcessPool
import traceback
from concurrent.futures import TimeoutError
import platform

from cmlkit import from_config, logger
from cmlkit.engine import compute_hash

from .resultdb import ResultDB


class EvaluationPool:
    """Wrapper around ProcessPool.

    Essentially a ProcessPool that can only evaluate configs,
    with some tweaks and caching.

    Aims to do the following:
        - Instantiate evaluators once in the workers to reduce overhead
        - Use a cache to not have to run evaluations twice
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
    to ever touch this class without `Run` mediating, I don't think this is a problem.

    But be careful out there.

    ***

    WARNING: macOS has some issues with multiprocessing and fork safety. This
    should not be a problem with this implementation, but if the models evaluated do
    something fancy, this might be the problem. So if you encounter something like
    `RuntimeError: Unexpected error within the Pool`, please check whether it persists
    on Linux. (I've observed this in particular with using something that relies on
    sqlite3 and attempts to write things to disk concurrently.)

    ---
    * We interface with external code that doesn't always play by the rules, and in
      particular is quite fond of not reacting to SIGTERM. The `concurrent.futures`
      `ProcessPoolExecutor` doesn't seem to be able to enforce a timeout in such cases.

    """

    def __init__(
        self,
        max_workers,
        evaluator_config,
        evaluator_context={},
        evals=None,
        trial_timeout=None,
        caught_exceptions=(TimeoutError,),
    ):

        self.trial_timeout = trial_timeout
        self.pool = ProcessPool(
            initializer=initializer,
            initargs=(evaluator_config, evaluator_context),
            max_workers=max_workers,
        )

        if platform.system() == "Darwin" and max_workers > 1:
            logger.warning("Parallel support on macOS is a bit wonky. Proceed with caution.")

        if evals is None:
            evals = ResultDB()

        self.evals = evals
        self.caught_exceptions = caught_exceptions

    def schedule(self, suggestion):
        """Schedule evaluation of a suggestion.

        This also checks the cache in the background, and creates a faux
        future to return the cached result. This is slightly inefficient,
        but it substantially reduces the complexity of the interface: We
        can now always expect a future as a result, and the re-submission
        can be handled in a unified way by the `Run`. (You can't simply
        keep requesting suggestions until you hit something is not in the
        cache, this leads to deadlocks when the search space has been exhausted.)
        """
        eid = compute_hash(suggestion)

        if eid in self.evals:
            result = self.evals.get_result(eid)
            future = self.pool.schedule(passthrough, args=(result,))
        else:
            future = self.pool.schedule(
                evaluate, args=(eid, suggestion), timeout=self.trial_timeout
            )

        future.eid = eid  # annotate with hash key in evals
        future.suggestion = suggestion

        return future

    def finish(self, future):
        """Obtain result of an evaluation future, catch errors, update caches.

        Should only be called with a finished future... but it's not a problem
        if it's not. The call to `future.result()` will trigger execution."""

        try:
            result = future.result()

            self.evals.submit_result(future.eid, result)
            return result
        except self.caught_exceptions as e:
            trace = traceback.format_exc()
            result = {
                "error": {
                    "error": e.__class__.__name__,
                    "error_text": str(e),
                    "traceback": trace,
                    "suggestion": future.suggestion,
                }
            }

            self.evals.submit_result(future.eid, result)
            return result
        except Exception as e:
            # uncaught exception, print suggestion and exit
            trace = traceback.format_exc()
            message = f"Unexpected error {e.__class__.__name__} evaluating a trial.\n"
            message += f"Error string: {e}.\n"
            message += f"Suggestion: {future.suggestion}.\n"
            message += f"Traceback:\n{trace}."

            logger.error(message)
            raise e

    def shutdown(self):
        self.pool.stop()  # no point in waiting for things
        try:
            self.pool.join(timeout=1.0)
            logger.info("Successfully and peacefully shut down pool.")
        except TimeoutError:
            logger.info("Failed to peacefully shut down pool... but no worries.")


def initializer(evaluator_config, evaluator_context):
    """Instantiate the evaluator once."""
    global evaluator
    evaluator = from_config(evaluator_config, evaluator_context)


def evaluate(eid, suggestion):
    """Actually perform the evaluation."""
    eval_result = evaluator(suggestion)
    eval_result["suggestion"] = suggestion

    return {"ok": eval_result}


def passthrough(result):
    """Simply return the result."""
    return result
