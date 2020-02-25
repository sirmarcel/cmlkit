"""Run.

This implements the Run class, with a bunch of helper classes.

Let's inventorise:
- exceptions: deals with de/serialisation of exceptions
- pool: wrapper for `pebble.ProcessPool`, handles evaluations in parallel
- resultdb: defines how results of trials/evaluations are stored
- run: actual run class
- state: keeps track of optimisation state
- stopping: stopping methods
- tape: how we save steps

Conventions:
- A *result* is a dict {"status": {# outcome}} where "status" is either "ok" or "error",
  and the outcome dict contains the actual results of a computation. (See `state.py` for
  the canonical definition.) This dict format is used internally to pass evaluation
  results around from the `pool` to the other components.


"""

from .run import Run
