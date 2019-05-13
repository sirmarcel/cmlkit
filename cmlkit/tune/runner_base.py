import os
import time
from copy import deepcopy
import logging
from datetime import datetime

import cmlkit as cml
from ..engine import Component, humanize, save_yaml, makedir


class RunnerBase(Component):
    """Runner prototype class"""

    kind = 'runner_base'

    # standalone: if True, we assume that we're performing a large-scale run,
    # and the Runner will write things to disk, print to stdout, etc.
    # if False, it will do its work, write only to the logger, and generally shut up.
    default_context = {'standalone': True, 'loglevel': 20}
    standalone_default_context = {'cache_type': 'mem+disk', 'basedir': os.getcwd()}

    def __init__(self, evaluator, search, name=None, evals={}, context={}):
        # hack to set a different default context if running in standalone mode,
        # because we want to set additional things in that case (like disk caching)
        standalone = context.get('standalone', self.__class__.default_context['standalone'])
        if standalone:
            context = {**self.__class__.standalone_default_context, **context}

        super().__init__(context=context)
        self.standalone = self.context['standalone']
        self.loglevel = self.context['loglevel']
        cml.logger.setLevel(self.loglevel)

        self.outdir = None
        self.top_n = 5  # how many models we will bother saving; TODO: make this configurable
        self.elapsed = 0.0  # elapsed time in run; should be updated within run()

        self.evaluator = cml.from_config(evaluator, context=context)
        self.evaluator_config = self.evaluator.get_config()

        self.search = cml.from_config(search, context=context)
        self.search_config = self.search.get_config()

        self.tid_to_eid = {}  # this maps search trial ids to evaluation ids
        self.cache_hits = 0

        self.task_hash = cml.engine.compute_hash(self.search_config, self.evaluator_config)

        if name is None:
            # just make sure we have something unique; otherwise we accidentally overwrite logs/status
            self.name = humanize(cml.engine.compute_hash(time.time(), self.search_config, self.evaluator_config), words=2)
        else:
            self.name = name

        if isinstance(evals, cml.Evals):
            self.evals = evals
        elif isinstance(evals, dict):
            # mainly for legacy reasons
            self.evals = cml.Evals(db=evals, normalise=True, name='evals')
        else:
            if os.path.isfile(evals):
                self.evals = cml.from_npy(evals)
            else:
                self.evals = cml.Evals(name='evals')

        # set up directory on disk
        if self.standalone:
            self.outdir = context['basedir'] + '/run_' + self.name
            might_overwrite = os.path.isdir(self.outdir)
            makedir(self.outdir)

            file_logger = logging.FileHandler(f"{self.outdir}/log.log")
            cml.logger.addHandler(file_logger)

            welcome = f"Set up {self.get_kind()} named {self.name} in standalone mode. Outdir is {self.outdir}. Loglevel is {self.loglevel}. Enjoy!"
            cml.logger.info(welcome)
            print(welcome)  # printing just in case something went wrong with the logs

            if might_overwrite:
                cml.logger.warn(f"Runner output directory {self.outdir} already exists, we might overwrite something.")

            save_yaml(self.outdir + '/search.yml', self.search_config)
            save_yaml(self.outdir + '/eval.yml', self.evaluator_config)
            save_yaml(self.outdir + '/runner.yml', self.get_config())

            self.history = self.outdir + f"/history.npy"  # the exact 'trajectory' of this run

        else:
            welcome = f"Set up {self.get_kind()} named {self.name} in interactive mode. Nothing will be saved!"
            cml.logger.info(welcome)

    def run(self, maxtime=120):
        raise NotImplementedError('Runners must implement run()!')

    def get_task(self):
        """Get a new evaluation task to run"""

        # search.suggest() is expected to return
        # (None, None) if it is currently
        # unwise to draw further samples.
        tid, config = self.search.suggest()

        if tid is not None:
            # eid identifies this evaluation of this model config
            # to the runner, as opposed to tid, which identifies trials
            # internally to the search algorithm
            eid = self._compute_eval_hash(config)

            self.tid_to_eid[tid] = eid

            return eid, config

        else:
            return None, None


    def _compute_eval_hash(self, config):
        return cml.engine.compute_hash(config)


    def in_cache(self, eid, config):
        # Use this in the single step of the
        # runner event loop, like so:
        # if self.in_cache(eid, config):
        #   pass
        # else:
        #   run_job(eid, config)
        if config is not None and eid in self.evals:
            cml.logger.debug(f"Checking cache for {eid}")
            self.cache_hits += 1
            # cache hit!
            # TODO: if the thing in the cache a) is a Timeout error
            # AND b) the timeout of this search instance (also TODO)
            # is bigger, pretend it's not in the cache and compute it!
            result = self.evals[eid]
            cml.logger.debug(f"Found eid {eid} in evals DB, submitting directly to search.")
            self.search.submit(self.eid_to_tid[eid], result)
            return True

        return False

    def submit_result(self, eid, result):
        result['config_eval'] = self.evaluator_config
        result['search_config'] = self.search_config
        result['run_name'] = self.name

        self.evals.submit(eid, result)
        self.search.submit(self.eid_to_tid[eid], result)

    def finish(self):
        # finishing up a run

        status = f"Finished running after {self.elapsed:.1f}s. If in standalone mode, I will save things, then exit. Thanks!"
        self.save_status(status)
        cml.logger.info(status)
        self.save()

        if self.standalone:
            # write out reports on the top_n models, for quick surveying of finished runs.
            result = 'Top 5 results:\n'
            for i, res in enumerate(self.best_n(self.top_n)):
                result += f"# Model Rank {i}: #" + '\n'
                report = res.get('report', 'no report available')
                result += report + '\n\n'

            with open(self.outdir + f"/results.txt", 'w') as f:
                f.write(result)

            status = f"Finished saving things into {self.outdir}. Exiting. Have a good day!"
            self.save_status(status)
            cml.logger.info(status)

        return self.best_n(self.top_n)


    def save(self):
        # save the current state of this run,
        # should not be called to often since it'll write not-tiny files
        self.save_evals()
        self.save_history()
        self.save_top_models()

    def save_evals(self):
        # should be called occasionally as the runner runs
        if self.standalone:
            self.evals.save(dirname=self.outdir)

    def save_top_models(self):
        # should be called occasionally as the runner runs
        if self.standalone:
            for i, res in enumerate(self.best_n(self.top_n)):
                cml.save_yaml(self.outdir + f"/model_{self.name}-{i}.yml", res['model_config'])

    def save_history(self):
        # should be called occasionally as the runner runs
        if self.standalone:
            to_save = {'losses': self.search.losses, 'eid_to_tid': self.eid_to_tid, 'suggestions': self.search.suggestions}
            cml.engine.safe_save_npy(self.history, to_save)

    def save_status(self, s):
        # Status is a short, overwritten text file that
        # can be conveniently catted/watched for progress
        if self.standalone:
            with open(self.outdir + f"/status.txt", 'w') as f:
                f.write(self.make_status(s))

    def make_status(self, s):
        # s should be a string with any info (multi-lines are fine)
        # that the particular Runner subclass wants to share with the user
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status = f"### Status of run {self.name} ({self.get_kind()}) at {time} ###\n"
        status += f"Best loss is {self.search.best_loss:.5f}; currently on trial {self.search.count} ({self.search.count_ok} ok, {self.search.count_error} error).\n"
        status += f"Trials since last improvement: {self.search.since_last_improvement}; Evals in db: {len(self.evals)} ({self.cache_hits} hits)\n"
        status += f"{s}\n\n"

        errors = self.evals.count_by_error()
        if len(errors) > 0:
            status += f"Error statistics:"
            for e, count in errors.items():
                status += f" {e[0]}: {count}"
            status += '\n'

        status += f"\nBest suggestion so far:\n"
        status += f"{self.search.best_suggestion}\n\n"

        return status

    def log_status(self, s):
        # should be called ~once per event
        cml.logger.info(self.make_logline(s))

    def make_logline(self, s):
        # s should be a short, single-line string that will be written to the log
        status = f"Best loss is {self.search.best_loss:.5f}; this is trial {self.search.count} ({self.search.count_ok} ok, {self.search.count_error} error). {s}"

        return status

    def best_n(self, n=3):
        """Return a list of best best n evals (reported by the search)"""
        tids_by_loss = self.search.tids_by_loss

        # make sure we don't crash in early stages of a run
        if len(tids_by_loss) < n:
            n = len(tids_by_loss)

        tids = tids_by_loss[0:n]
        eids = [self.tid_to_eid[tid] for tid in tids]

        return [self.evals[eid] for eid in eids]

    @property
    def eid_to_tid(self):
        return {v: k for k, v in self.tid_to_eid.items()}
