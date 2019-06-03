"""Implement stopping methods."""

from cmlkit.engine import Configurable


class StopMax(Configurable):
    """Stopping by maximum trials or evaluations.

    Parameters:
        count: Integer, maximum count
        errors: Boolean, whether to count errors
        use: either "trials" or "evals". which results
            to count. Evals only counts unique suggestions
            that were tried, while trials counts everything
            the search suggested, including repeats.
            using "evals" with a small search space can result
            in deadlock, since it might be possible that not enough
            viable candidates exist.

    """

    kind = "stop_max"

    def __init__(self, count, errors=True, use="trials"):
        self.count = count
        self.errors = errors

        assert use in ["trials", "evals"]
        self.use = use

    def done(self, state):
        return self.compute_count(state) >= self.count

    def compute_count(self, state):
        if self.use == "evals":
            results = state.evals
        elif self.use == "trials":
            results = state.trials

        if self.errors:
            return len(results)
        else:
            return len(results.where_state("ok"))

    def short_report(self, state):
        return f"Counted {self.use}: {self.compute_count(state)}/{self.count}."

    def _get_config(self):
        return {"count": self.count, "errors": self.errors, "use": self.use}


classes = {StopMax.kind: StopMax}
