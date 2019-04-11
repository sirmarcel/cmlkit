import numpy as np

from cmlkit import load_dataset
from .evaluator_base import EvaluatorBase


class EvaluatorCV(EvaluatorBase):
    """Evaluate model on splits of one dataset"""

    kind = 'eval_cv'

    default_context = {'prime_datasets': True}

    def __init__(self, data, idx, target,
                 predict_per=None,
                 lossf='pred',
                 include_var=True,
                 sort=True,
                 context={}):
        super().__init__(target, predict_per=predict_per, lossf=lossf, context=context)

        # if true, we will call .incidence, .info and .mask on all datasets/
        # Views, ensuring that they only have to be called once.
        self.prime_datasets = self.context['prime_datasets']

        self.data = load_dataset(data)
        if self.prime_datasets:
            self.data.info

        self.include_var = include_var
        self.sort = sort

        if self.sort:
            self.idx = [[np.sort(np.array(train, dtype=int)), np.sort(np.array(test, dtype=int))] for train, test in idx]
        else:
            self.idx = [[np.array(train, dtype=int), np.array(test, dtype=int)] for train, test in idx]

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {'data': self.data.name,
                'idx': [[train.tolist(), test.tolist()] for train, test in self.idx],
                'target': self.target,
                'include_var': self.include_var,
                'predict_per': self.predict_per,
                'sort': self.sort,
                'lossf': [l.name for l in self.lossf],
                }

    def _evaluate(self, model):
        loss = model.cv_losses(self.data, self.idx, self.target, per=self.predict_per, lossfs=self.lossf)

        report = self.make_report(model, loss)

        var = {}
        if self.include_var and 'cv' in loss:
            var = {k + '_var': v['var'] for k, v in loss['cv'].items()}

        result = {**loss,
                  **var,
                  'report': report, }

        return result

    def make_report(self, model, loss):

        report = '# CV Losses #\n\n'
        report += f"Master dataset was {self.data.name} (property {self.target}/{model.predict_per}) \n"
        report += f"Predicted on {len(self.idx)} splits (property {self.target}/{self.predict_per})\n"

        report += self._format_loss(loss) + '\n'

        return report

    def _format_loss(self, loss):
        report = ''
        if self.includes_defaults:
            report += f"  => RMSE={loss['rmse']:.3f} MAE={loss['mae']:.3f} R2={loss['r2']:.3f}\n\n"

        for k, v in loss.items():
            if k != 'cv':
                report += f"     {k.upper()}={v}\n"

        if 'cv' in loss:
            report += '\n     Detailed report:\n\n'
            for k, v in loss['cv'].items():
                report += f"     {k.upper()}: mean: {v['mean']:.3f}, std: {v['var']:.6f}, var: {v['var']:.6f}\n"
                report += f"     all: {v['losses']}\n\n"

        return report
