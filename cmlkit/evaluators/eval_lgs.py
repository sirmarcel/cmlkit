from copy import deepcopy
import traceback

from cmlkit import logger, classes
from ..engine import Component, _from_config, read_yaml


class EvaluatorLGS(Component):
    """Evaluate model with a (local) grid search performed on a target quantity

    Note that this should be used as part of a hyper-parameter searching pipeline,
    not as an 'actual' evaluation, where optimising for the test set is, of course,
    an utterly terrible idea! This class is intended to separate the fine-tuning
    of numerical parameters from a more coarse-grained global parameter search.
    """

    kind = 'eval_lgs'

    default_context = {'cache_type': 'mem+disk', 'print_timings': False}

    def __init__(self, evaluator, lgs={'kind': 'lgs', 'config': {}}, loss='rmse', context={}):
        super().__init__(context=context)
        self.evaluator = evaluator
        self.lgs = lgs
        self.loss = loss
        assert isinstance(loss, str), 'EvaluatorLGS requires a string name for the loss!'

        self.evaluator = cml._from_config(evaluator, context=self.context, classes=classes)
        self.lgs = cml._from_config(lgs, context=self.context, classes=classes)

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {'loss': self.loss,
                'evaluator': self.evaluator.get_config(),
                'lgs': self.lgs.get_config()}

    def evaluate(self, model):
        config = self._get_model_config(model)
        original_config = deepcopy(config)

        if not self.lgs.needs_lgs(config):
            return self.evaluator.evaluate(config)  # passthrough in case no optimisation is needed

        try:
            lgs_info = self._optimize(config)
        except Exception as e:
            # TODO: Use more general form from base class
            trace = traceback.format_exc()
            cml.logger.error(f"An error occurred while running lgs on model: {e}. Model config (at time of error): {config}. \n {trace}")
            result = {'status': 'error',
                      'error': (e.__class__.__name__, str(e)),
                      'report': f"ERROR during LGS: {e}",
                      'eval_config': self.get_config(),
                      'model_config': original_config,
                      'traceback': trace}

            return result

        # final evaluation, which is a bit redundant but neat
        result = self.evaluator.evaluate(config)

        result['config_lgs_eval'] = self.get_config()
        result['config_before_lgs'] = original_config
        result['report'] += f"\nRan LGS (loss {self.loss}) for the following variables: {lgs_info['locations']} in {lgs_info['duration']:.1f}s\n\n"

        return result

    def _get_model_config(self, model):
        # we need to make sure that we have a config dict,
        # so LGS has something to parse!
        if isinstance(model, Component):
            config = model.get_config()
        elif isinstance(model, str):
            config = cml.read_yaml(model)
        elif isinstance(model, dict):
            config = deepcopy(model)
        else:
            raise ValueError(f"Do not know how to evaluate {model}.")

        return config

    def _optimize(self, config):
        def target(this_config):
            this_model = cml._from_config(this_config, context=self.context, classes=classes)
            loss = self.evaluator._evaluate(this_model)
            return loss[self.loss]

        info = self.lgs.optimize(config, target)

        if self.context['print_timings']:
            print(f"Finished LGS in {info['duration']:.1f}s.")

        return info
