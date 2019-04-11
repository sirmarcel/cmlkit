import os
import time
from copy import deepcopy
import concurrent.futures

from cmlkit import logger
from .runner_base import RunnerBase


class RunnerPool(RunnerBase):
    """Search runner using concurrent.futures.ProcessPoolExecutor as parallelisation backend."""

    kind = 'runner_pool'

    def __init__(self, evaluator, search, nworkers=4, name=None, wait_per_loop=10.0, evals={}, context={}):
        self.nworkers = nworkers

        # number of seconds to wait for new results per loop;
        # this is just to prevent the loop from getting completely
        # stuck if evaluations take a long time, this way we at least
        # get to regularly report status + save the eval db
        self.wait_per_loop = wait_per_loop

        super().__init__(evaluator, search, name=name, evals=evals, context=context)

        # max_new_evals is the max number of new evaluations
        # that are submitted at once to the pool during one
        # step in the main event loop.
        # It can't be 1 because then the pool doesn't get refilled
        # if the tasks complete faster than the event loop;
        # it also can't be infinity, because then we might submit tasks
        # that are not necessary (since the search has not obtained any
        # new information)
        if hasattr(self.search, 'max_concurrent'):
            self.max_new_evals = min(self.search.max_concurrent, self.nworkers) + 1
        else:
            self.max_new_evals = self.nworkers + 1

    @classmethod
    def _from_config(cls, config, context={}):
        defaults = {'name': None, 'evals': {}, 'nworkers': 4, 'wait_per_loop': 10.0}
        c = {**defaults, **config}
        return cls(c['evaluator'], c['search'],
                   nworkers=c['nworkers'],
                   name=c['name'],
                   wait_per_loop=c['wait_per_loop'],
                   evals=c['evals'],
                   context=context)

    def _get_config(self):
        return {'evaluator': self.evaluator_config,
                'search': self.search_config,
                'name': self.name,
                'nworkers': self.nworkers,
                'wait_per_loop': self.wait_per_loop}

    def run(self, timeout=None):  # slightly short of 24h
        start = time.time()
        self.save()
        last_save = 0.0
        since_save = 0.0

        last_update = -60.0  # force immediate update on start!
        futures = {}

        # warm start
        while not self.search.done:
            eid, config = self.get_task()
            if not self.in_cache(eid, config) or eid is None:
                break

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nworkers) as main_exec:  # this pool is for evaluation work
            with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='cml_io_') as aux_exec:  # this pool is for i/o, like saving the eval db
                while not self.search.done:
                    self.elapsed = time.time() - start
                    since_save = self.elapsed - last_save

                    if timeout is not None:
                        if self.elapsed > timeout:
                            status = f"Breaking due to timeout (will wait up to {self.wait_per_loop}s to finish computations)."
                            self.save_status(status)
                            logger.info(status)
                            break

                    # log every 2s to avoid congesting i/o when progressing quickly
                    if self.elapsed - last_update > 2:
                        status = f"Elapsed time {self.elapsed:.1f}s; Running tasks: {len(futures)}; Last save: {since_save:.1f}s ago."
                        self.log_status(status)
                        self.save_status(status)
                        last_update = self.elapsed

                    # dispatch saving task every 60s
                    if since_save > 60:
                        aux_exec.submit(self.save)
                        last_save = self.elapsed

                    # obtain finished evaluations
                    done, not_done = concurrent.futures.wait(futures, timeout=self.wait_per_loop,
                                                             return_when=concurrent.futures.FIRST_COMPLETED)

                    # submit finished evaluations to search
                    for finished in done:
                        eid = futures[finished]
                        res = finished.result()

                        self.submit_result(eid, res)
                        del futures[finished]

                    # refill the pool with new tasks
                    # TODO: consider staggering these submissions on starting a completely
                    # new run of a search so the caches don't overwrite each other
                    for i in range(abs(len(futures) - self.max_new_evals) + 1):
                        eid, config = self.get_task()
                        logger.debug(f"Refilling task {i} with eid {eid}.")

                        if not self.in_cache(eid, config) and config is not None:
                            f = main_exec.submit(self.evaluator.evaluate, config)
                            futures[f] = eid

                status = f"Finished main loop, waiting for executors to wrap up."
                self.log_status(status)
                self.save_status(status)

        self.finish()
