import traceback
import cmlkit as cml
from cmlkit.engine import Component


class EvaluatorBase(Component):
    """Base class for evaluators"""

    kind = 'eval_base'

    def __init__(self, target, predict_per=None, lossf='pred', context={}):
        super().__init__(context=context)

        self.target = target
        self.predict_per = predict_per

        self.lossf = cml.get_loss(lossf)
        assert isinstance(self.lossf, list), "Evaluators expect to compute lists of losses only (use plain Model instances for single losses :)"

        lossnames = [l.name for l in self.lossf]
        self.includes_defaults = all(x in lossnames for x in ['rmse', 'mae', 'r2'])  # just check whether we have the usual suspect on hand

    def evaluate(self, model):
        model, model_config = self._get_model(model)

        eval_config = self.get_config()

        try:
            result = self._evaluate(model)
            return {**result,
                    'status': 'ok',
                    'eval_config': eval_config,
                    'model_config': model_config}

        except Exception as e:
            # TODO: Should I only capture predefined errors?
            trace = traceback.format_exc()
            cml.logger.error(f"An error occurred while evaluating model: {e}. Model config: {model_config}. \n {trace}")
            result = {'status': 'error',
                      'error': (e.__class__.__name__, str(e)),
                      'report': f"ERROR: {e}",
                      'eval_config': eval_config,
                      'model_config': model_config,
                      'traceback': trace}

            return result

    def _get_model(self, model):
        if isinstance(model, dict):
            model = cml.from_config(model, context=self.context)
        if isinstance(model, str):
            model = cml.from_yaml(model, context=self.context)

        model_config = model.get_config()
        return model, model_config
