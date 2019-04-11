import time
from copy import deepcopy
import cmlkit2 as cml2
import concurrent.futures
import os
from cmlkit2.tune import RunnerBase


class RunnerSingle(RunnerBase):
    """Search runner for single core only (very basic)"""

    kind = 'runner_single'

    def __init__(self, evaluator, search, name=None, evals={}, context={}):
        super().__init__(evaluator, search, name=name, evals=evals, context=context)

    @classmethod
    def _from_config(cls, config, context={}):
        defaults = {'name': None, 'evals': {}}
        c = {**defaults, **config}
        return cls(c['evaluator'], c['search'], name=c['name'], evals=c['evals'], context=context)

    def _get_config(self):
        return {'evaluator': self.evaluator_config,
                'search': self.search_config,
                'name': self.name}

    def step(self):
        eid, config = self.get_task()

        if self.in_cache(eid, config):
            # this will submit cached value to search,
            # then go to the next iteration
            pass
        elif config is not None:
            result = self.evaluator.evaluate(config)
            self.submit_result(eid, result)
        elif config is None:
            cml2.logger.info('Received None as suggestion, waiting for a bit.')
            time.sleep(0.5)

    def run(self, timeout=None):
        start = time.time()
        last_save = 0.0

        # we're only using this pool to do asynchronous saving of the evals
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while not self.search.done:
                self.elapsed = time.time() - start
                status = f"Elapsed time {self.elapsed:.1f}. Last save at t={last_save:.1f}."
                self.log_status(status)

                # saving is done asynchronously
                if self.elapsed - last_save > 25:
                    executor.submit(self.save_status, status)
                    executor.submit(self.save_evals)
                    last_save = self.elapsed

                if timeout is not None:
                    if self.elapsed > timeout:
                        self.save_status('Timeout')
                        cml2.logger.info('Timeout!')
                        break

                self.step()

        self.finish()
