"""Run.

Run actually performs hyper-parameter tuning,
combining a `Search`, which provides the suggestions and an
`Evaluator`, which defines how losses are computed.

The actual event loop and execution happens in a
ProcessPool provided by `pebble`.

Run also takes care of saving things to disk, resuming
itself, stuff like that.

This is the kitchen sink, in other words.

"""

from concurrent.futures import TimeoutError, wait, FIRST_COMPLETED
from multiprocessing import cpu_count
import traceback
import time
from pathlib import Path
import numpy as np
from datetime import datetime
from logging import FileHandler, INFO
import textwrap

from cmlkit.engine import Component, compute_hash
from cmlkit import from_config, logger
from cmlkit.engine import to_config, _from_config, makedir, save_yaml
from cmlkit.utility import humanize
from cmlkit.env import get_scratch

from .stopping import classes as stoppers
from .exceptions import get_exceptions, get_exceptions_spec
from .pool import EvaluationPool
from .resultdb import ResultDB
from .state import State
from .tape import Tape


class Run(Component):

    kind = "run"

    default_context = {
        "max_workers": cpu_count(),
        "shutdown_duration": 30.0,
        "wait_per_loop": 5.0,
    }

    def __init__(
        self,
        search,
        evaluator,
        stop,
        trial_timeout=None,
        caught_exceptions=["TimeoutError"],
        name=None,
        context={},
    ):
        super().__init__(context=context)

        self.search = from_config(search)
        self.evaluator_config = to_config(evaluator)
        self.stop = _from_config(stop, classes=stoppers)
        self.trial_timeout = trial_timeout
        self.caught_exceptions = get_exceptions(caught_exceptions)

        self.id = compute_hash(time.time(), np.random.random(10))  # unique id of this run

        if name is None:
            self.name = humanize(self.id, words=2)
        else:
            self.name = name

        self.ready = False  # run is not ready until prepared
        self.readonly = False  # only True if instance is obtained through Run.checkout()
        # (which is the "inspect result of run" mode, and not intended to yield something
        # that can be continued)

    def _get_config(self):
        return {
            "search": self.search.get_config(),
            "evaluator": self.evaluator_config,
            "stop": self.stop.get_config(),
            "trial_timeout": self.trial_timeout,
            "caught_exceptions": get_exceptions_spec(self.caught_exceptions),
            "name": self.name,
        }

    def prepare(self, directory=Path(".")):
        assert not self.readonly, "Run cannot be prepared in read only mode."

        work_directory = directory / f"run_{self.name}"
        makedir(work_directory)

        evals = ResultDB()
        tape = Tape.new(metadata=self.get_config(), filename=work_directory / "tape.son")

        state = State(search=self.search, evals=evals, tape=tape)

        self._prepare(work_directory, evals, state)

    def _prepare(self, work_directory, evals, state, msg="Prepared"):
        self.work_directory = work_directory

        self.pool = EvaluationPool(
            evals=evals,
            max_workers=self.context["max_workers"],
            evaluator_config=self.evaluator_config,
            evaluator_context=self.context,
            trial_timeout=self.trial_timeout,
            caught_exceptions=self.caught_exceptions,
        )
        self.state = state

        logger.addHandler(FileHandler(f"{self.work_directory}/log.log"))
        logger.setLevel(INFO)
        logger.info(f"{msg} runner {self.name} in folder {self.work_directory}.")

        self.ready = True

    @classmethod
    def restore(cls, directory, new_stop=None, context={}):
        logger.info("Starting run restore...")

        directory = Path(directory)
        backup_tape = directory / "bak-tape.son"
        (directory / "tape.son").rename(backup_tape)
        old_tape = Tape.restore(backup_tape)

        runner_config = old_tape.metadata["run"]
        if new_stop is not None:
            runner_config["stop"] = to_config(new_stop)

        run = Run.from_config(runner_config, context=context)

        new_tape = Tape.new(metadata=run.get_config(), filename=directory / "tape.son")

        evals = ResultDB()
        state = State.from_tape(
            search=run.search, tape=old_tape, evals=evals, new_tape=new_tape
        )  # now we have successfully replayed the optimisation so far

        run._prepare(directory, evals, state, msg="Recovered")

        return run

    @classmethod
    def checkout(cls, directory):
        """Get read-only run from directory.

        This is the canonical way of obtaining the results of a
        finished (or, inadvisably, an ongoing) run. It returns a
        Run instance that cannot be run(), but is otherwise in the
        same state as the run that is being checked out.

        """
        directory = Path(directory)

        logger.info("Starting run checkout... (this will not yield a runnable instance).")

        original_tape = Tape.restore(directory / "tape.son")
        run = Run.from_config(original_tape.metadata["run"])
        state = State.from_tape(search=run.search, tape=original_tape)

        run.state = state
        run.readonly = True

        return run

    def __call__(self, duration=float("inf")):
        """Run for duration minutes."""

        return self.run(duration=duration)

    def run(self, duration=float("inf")):
        """Run for duration minutes."""

        assert self.ready, "prepare() must be called before starting run."

        duration = (duration * 60.0) - self.context["shutdown_duration"]

        start = time.monotonic()
        end = start + duration

        futures = {}

        # when recovering, first re-submit the running trials
        if len(self.state.live_trials) > 0:
            logger.info(
                f"Re-submitting {len(self.state.live_trials)} trials to the pool."
            )
            for tid, suggestion in self.state.live_trials.items():
                f = self.pool.schedule(suggestion)
                futures[f] = tid

        while time.monotonic() < end and not self.stop.done(self.state):
            status = self.write_status(
                "Running.", len(futures), time.monotonic() - start, duration
            )
            logger.info(status)

            done, running = wait(
                futures,
                timeout=self.context["wait_per_loop"],
                return_when=FIRST_COMPLETED,
            )

            if not self.stop.done(self.state):
                for f in done:
                    tid = futures[f]
                    result = self.pool.finish(f)
                    self.state.submit(tid, result)
                    del futures[f]

                n_new_trials = max(0, self.context["max_workers"] + 1 - len(running))
                for i in range(n_new_trials):
                    tid, suggestion = self.state.suggest()
                    f = self.pool.schedule(suggestion)

                    futures[f] = tid

        runtime = time.monotonic() - start
        logger.info(f"Finished run {self.name} in {runtime:.2f}s. Starting shutdown...")
        self.write_status(f"{self.name}: Done, saving results.", 0, runtime, duration)
        self.write_results()
        self.write_status(
            f"{self.name}: Done, initiating shutdown.", 0, runtime, duration
        )
        self.pool.shutdown()
        self.write_status(f"{self.name}: Done. Have a good day!", 0, runtime, duration)

    def write_results(self):
        for i, config in enumerate(self.state.evals.top_suggestions()):
            save_yaml(self.work_directory / f"suggestion-{i}", config)
        logger.info(f"Saved top 5 suggestions into {self.work_directory}.")

        refined = self.state.evals.top_refined_suggestions()
        if any([r != {} for r in refined]):
            for i, config in enumerate(self.state.evals.top_refined_suggestions()):
                save_yaml(self.work_directory / f"refined_suggestion-{i}", config)
            logger.info(f"Saved top 5 refined suggestions into {self.work_directory}.")

    def write_status(self, message, n_futures, runtime, duration):
        timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"### Status of run {self.name} at {timestr} ###\n"
        status = f"{message} Runtime: {runtime:.1f}/{duration:.1f}. Active evaluations: {n_futures}."
        body = "\n".join(
            [status, self.stop.short_report(self.state), self.state.short_report()]
        )
        full_status = header + textwrap.indent(body, " ")

        with open(self.work_directory / "status.txt", "w+") as f:
            f.write(full_status + "\n")

        return full_status
