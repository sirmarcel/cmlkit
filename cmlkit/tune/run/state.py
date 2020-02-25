"""Run/search state management."""

from copy import deepcopy
from cmlkit.engine import parse_config, compute_hash

from .resultdb import ResultDB
from .tape import Tape


class State:
    """Tracks the (optimisation) state of a run.

    In essence, this object holds all the information that
    defines what the optimisation is doing at this current step,
    provides the `suggest` and `submit` methods to advance the state,
    and provides a simple, but powerful way of recovering state:

    We simply record every action taken during the course of the
    optimisation, and to recover state we simply redo all previous steps.
    This saves us the pain of trying to dump all the various moving parts
    of a run, some of which have no obvious or robust serialisation methods.
        (Sideeye to hyperopt.)
    Since we can write this ongoing "tape" to disk in append mode, it
    is pretty robust to the process getting suddenly killed (Sideeye to slurm.)

    In addition to this bookkeeping work, this class is also the container that
    combines all the various stateful (but not concerned with execution) bits of
    a Run in a central location, and which can then be consumed by reporting tools
    and the various stopping methods.

    """

    def __init__(self, search, evals=None, tape=None):
        self.search = search

        if evals is None:
            self.evals = ResultDB()  # evaluations cache, see pool for more
            # only really here because we need to repopulate it when replaying.
        else:
            self.evals = evals

        if tape is None:
            tape = Tape.new()
        self.tape = tape
        self.trials = ResultDB()  # where we store results of trials

        self.live_trials = {}  # needed to restart!

    def record(self, action, payload):
        self.tape.append({action: deepcopy(payload)})

    def suggest(self):
        tid, suggestion = self.search.suggest()
        self.record("suggest", {"tid": tid, "suggestion": suggestion})

        self.live_trials[tid] = suggestion

        return tid, suggestion

    def submit(self, tid, result):
        state, outcome = check_result(result)

        self.search.submit(tid, for_search(state, outcome))
        self.record("submit", {"tid": tid, "result": result})

        self.trials.submit(tid, state, outcome)

        del self.live_trials[tid]

    @classmethod
    def from_tape(cls, tape, search, evals=None, new_tape=None):
        state = cls(search=search, evals=evals, tape=new_tape)
        state.replay(tape)

        return state

    def replay(self, tape):
        for record in tape:
            action, payload = parse_config(record)

            if action == "suggest":
                self.replay_suggest(payload)
            if action == "submit":
                self.replay_submit(payload)

    def replay_suggest(self, payload):
        tid, suggestion = self.suggest()
        assert (
            tid == payload["tid"]
        ), f"Replay failed because tid {tid} didn't match value on tape {payload['tid']}"
        assert (
            suggestion == payload["suggestion"]
        ), f"Replay failed because suggestion {suggestion} didn't match value on tape {payload['suggestion']}"

    def replay_submit(self, payload):
        tid = payload["tid"]
        result = payload["result"]
        self.submit(tid, result)

        # refilling the evals...! (since the pool can't do it)
        state, outcome = parse_config(result)
        eid = compute_hash(outcome["suggestion"])
        self.evals.submit_result(eid, result)

    def short_report(self):
        """Return a short overview of current state."""
        loss = "Best 3:"
        for l in self.evals.top_losses(3):
            loss += f" {l:.4f}"
        loss += "."
        counts = f"Live: {len(self.live_trials)}/T: {len(self.trials)} ({len(self.trials.where_state('ok'))})/E: {len(self.evals)} ({len(self.evals.where_state('ok'))})."
        state = " ".join([loss, counts])

        errors = self.trials.count_by_error()
        if errors != {}:
            state = "\n".join([state, str(errors)])

        return state


def check_result(result):
    """Enforce the run-internal result format."""

    state, outcome = parse_config(result)

    if state not in ["error", "ok"]:
        raise ValueError(f"Received a result with invalid state={state}.")

    if state == ["ok"]:
        if "loss" not in outcome:
            raise ValueError(f"Results with status 'ok' must contain a loss.")
        if "suggestion" not in outcome:
            raise ValueError(f"Results with status 'ok' must contain a suggestion dict.")

    return state, outcome


def for_search(state, outcome):
    """Format evaluation result for consumption by search."""

    if state == "error":
        return {"error": True}
    if state == "ok":
        loss = outcome["loss"]
        var = outcome.get("var", None)
        return {"loss": loss, "var": var}
