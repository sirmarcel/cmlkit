import cmlkit as cml
from .search_base import SearchBase


class SearchFixed(SearchBase):
    """Search a fixed set of candidates for the best one"""

    kind = 'search_fixed'

    def __init__(self, candidates, loss='rmse', context={}):

        space = []
        for c in candidates:
            if isinstance(c, dict):
                space.append(c)
            elif isinstance(c, str):
                space.append(cml.read_yaml(c))
            else:
                space.append(c.get_config())

        super().__init__(space, loss=loss, maxevals=1000000, seed=1, context=context)

        self.iterator = iter(self.space)

    @classmethod
    def _from_config(cls, config, context={}):
        defaults = {'loss': 'rmse'}
        config = {**defaults, **config}
        return cls(config['space'],
                   loss=config['loss'])

    def _get_config(self):
        return {
            'space': self.space,
            'loss': self.loss
        }

    def _suggest(self):

        tid = self._generate_tid()
        try:
            suggestion = next(self.iterator)
        except StopIteration:
            cml.logger.info('We have looked at all candidates.')
            self._done = True
            return None, None

        return tid, suggestion

    def _submit(self, tid, loss, error=None, var=None):
        # we don't care
        pass
