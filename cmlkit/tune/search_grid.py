import random
from copy import deepcopy
import itertools
import numpy as np
import time

from cmlkit import logger
from ..helpers import find_pattern, get_with_path, set_with_path
from .search_base import SearchBase


class SearchGrid(SearchBase):
    """Test implementation of grid search

    Expects space to be the dictionary to optimise
    over, where the grids to explore are marked by
    ['gs', [list of choices]].

    If we actually use this, it'll be good to implement
    automatically setting up loggrids and stuff, but who cares.

    """

    kind = 'search_grid'

    def __init__(self, space, loss='rmse', maxevals=25, shuffle=False, seed=1, context={}):
        super().__init__(space, loss=loss, maxevals=maxevals, seed=seed, context=context)

        self.variables, self.locations = self._parse_space(deepcopy(self.space))

        self.iterator = itertools.product(*self.variables)

    def _get_config(self):
        return {
            'maxevals': self.maxevals,
            'seed': self.seed,
            'space': self.space,
            'loss': self.loss
        }

    def _parse_space(self, space):

        def gs_pattern(x):
            if isinstance(x, (tuple, list)):
                return x[0] == 'gs'

            return False

        locations = find_pattern(space, gs_pattern)
        variables = [get_with_path(space, loc)[1] for loc in locations]

        return variables, locations

    def _make_suggestion(self, choices):
        suggestion = deepcopy(self.space)

        for i, val in enumerate(choices):
            set_with_path(suggestion, self.locations[i], val)

        return suggestion

    def _suggest(self):

        tid = self._generate_tid()
        try:
            choices = next(self.iterator)
        except StopIteration:
            logger.info('Grid search has exhausted the search space.')
            self._done = True
            return None, None

        suggestion = self._make_suggestion(choices)

        return tid, suggestion

    def _submit(self, tid, loss, error=None, var=None):
        # grid search does not care about anything
        # it gets back!
        pass
