"""Tools for hyperparameter tuning.

The basic idea is this: We separate out the optimisation process
into something that makes suggestions where to look next (Search),
something that evaluates (Evaluator) and something that takes care
of retrieving suggestions from the Search, running the Evaluator and
then feeding the results back into the Search, with various backends
for performance and parallelisation: The Runners.

Runners will usually maintain a database of known evaluations,
in order to be able to quickly restart searches.

Basic conventions:

Terminology:
suggestion -- dict describing a particular model
evaluation -- task of evaluating a config with the evaluator
eid -- id of an evaluation (used to save them to disk)
trial -- internally, data point (i.e. suggestion + its evaluation) that the search knows about
tid -- id of a trial, used by the search to track it

Evaluators consume config-style dicts and return hyperopt-style result
dicts, which should contain at least {'loss': 123, 'status': 'ok'/'error'}.

If an error occurs, the Evaluator should catch it and return an additional key
'error': (Exception, 'string of exception')

They are encouraged to compute a bunch of losses at once, since once a model
is trained and has predicted something, these are almost free to compute.
Since results are typically saved in a database by Runners, this allows us to
re-run optimisations for different losses for essentially free.

"""

# from .evals import Evals
# from .runner_base import RunnerBase
# from .runner_single import RunnerSingle
# from .runner_pool import RunnerPool
# from .search_base import SearchBase
# from .search_grid import SearchGrid
# from .search_hyperopt import SearchHyperopt
# from .search_fixed import SearchFixed
# from .opt_lgs import OptimizerLGS

components = []
