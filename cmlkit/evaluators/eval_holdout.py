import cmlkit2 as cml2
from cmlkit2.evaluators import EvaluatorBase


class EvaluatorHoldout(EvaluatorBase):
    """Evaluate model on a holdout set"""

    kind = 'eval_holdout'

    def __init__(self, data_train, data_test, target,
                 predict_per=None,
                 lossf='pred',
                 include_training=False,
                 include_nonconvert=False,
                 include_pred=False,
                 context={}):
        super().__init__(target, predict_per=predict_per, lossf=lossf, context=context)

        self.include_training = include_training
        self.include_nonconvert = include_nonconvert
        self.include_pred = include_pred  # whether to include full predictions/true values in result

        self.data_train = cml2.load_dataset(data_train)
        self.data_test = cml2.load_dataset(data_test)

    @classmethod
    def _from_config(cls, config, context={}):
        return cls(**config, context=context)

    def _get_config(self):
        return {'data_train': self.data_train.name,
                'data_test': self.data_test.name,
                'target': self.target,
                'predict_per': self.predict_per,
                'lossf': [l.name for l in self.lossf],
                'include_training': self.include_training,
                'include_nonconvert': self.include_nonconvert,
                'include_pred': self.include_pred, }

    def _evaluate(self, model):
        model.train(self.data_train, target=self.target)
        loss = model.loss(self.data_test, target=self.target, per=self.predict_per, lossf=self.lossf)

        training_loss = None
        nonconvert_loss = None
        true = None
        pred = None

        if self.include_training:
            training_loss = model.loss(self.data_train, target=self.target, per=self.predict_per, lossf=self.lossf)

        if self.include_nonconvert:
            nonconvert_loss = model.loss(self.data_train, target=self.target, lossf=self.lossf, per='original')

        if self.include_pred:
            pred = model.predict(self.data_test, target=self.target, per=self.predict_per)
            true = self.data_test.pp(self.target, per=self.predict_per)

        report = self.make_report(model, loss, training_loss, nonconvert_loss)

        result = {**loss,
                  'nonconvert_loss': nonconvert_loss,
                  'training_loss': training_loss,
                  'report': report,
                  'true': true,
                  'pred': pred}

        return result

    def make_report(self, model, loss, training_loss, nonconvert_loss):

        report = '# Losses #\n\n'
        report += f"Trained model on {self.data_train.name} (property {self.target}/{model.predict_per}) \n"
        report += f"Predicted on {self.data_test.name} (property {self.target}/{self.predict_per})\n"

        report += self._format_loss(loss) + '\n'

        if self.include_training:
            report += '\n# Training set losses\n\n'
            report += f"Trained model on {self.data_train.name} (property {self.target}/{model.predict_per}) \n"
            report += f"Predicted on {self.data_train.name} (property {self.target}/{self.predict_per})\n"

            report += self._format_loss(training_loss) + '\n'

        if self.include_nonconvert:
            report += '\n# Unconverted losses\n\n'
            report += f"Trained model on {self.data_train.name} (property {self.target}/{model.predict_per}) \n"
            report += f"Predicted on {self.data_test.name} (property {self.target}/{model.predict_per})\n"

            report += self._format_loss(nonconvert_loss) + '\n'

        return report

    def _format_loss(self, loss):
        report = ''
        if self.includes_defaults:
            report += f"  => RMSE={loss['rmse']:.3f} MAE={loss['mae']:.3f} R2={loss['r2']:.3f}\n\n"

        for k, v in loss.items():
            report += f"     {k.upper()}={v}\n"

        return report
