import numpy as np
import time
import cmlkit as cml
from .engine import Component


class Model(Component):
    """Model"""

    default_context = {
        'print_timings': False,
        'use_naive_cv': False,  # if True, cv routines offered by regressor will be ignored
    }

    def __init__(self, representation, regressor, predict_per=None, context={}):
        super().__init__(context=context)

        self.print_timings = self.context['print_timings']
        self.use_naive_cv = self.context['use_naive_cv']

        if isinstance(representation, dict):
            # we got a config, let us instantiate the representation
            self.representation = cml.from_config(representation, context=self.context)
        elif isinstance(representation, (list, tuple)):
            # we got a composed rep (or garbage!)
            self.representation = cml.ComposedRepresentation(*representation, context=self.context)
        elif isinstance(representation, Component):
            self.representation = representation
        else:
            raise ValueError('Cannot do anything with representation {}'.format(representation))

        if isinstance(regressor, dict):
            # we got a config, let us instantiate the regressor
            self.regressor = cml.from_config(regressor, context=self.context)
        elif isinstance(regressor, Component):
            self.regressor = regressor
        else:
            raise ValueError('Cannot do anything with regressor {}'.format(representation))

        self.predict_per = predict_per

    @classmethod
    def _from_config(cls, config, context={}):
        if 'predict_per' in config:
            return cls(config['representation'], config['regressor'], predict_per=config['predict_per'], context=context)
        else:
            return cls(config['representation'], config['regressor'], context=context)

    def _get_config(self):
        return {'representation': self.representation.get_config(),
                'regressor': self.regressor.get_config(),
                'predict_per': self.predict_per}

    def train(self, data, target):
        n = data.n

        start = time.time()
        self.x_train = self.representation.compute(data)
        if self.print_timings:
            cml.logger.info('Computed training representation in {:.2f}s (n={}).'.format(time.time() - start, n))

        start = time.time()
        self.y_train = cml.convert(data, data.p[target], self.predict_per)

        self.regressor.train(self.x_train, self.y_train)

        if self.print_timings:
            cml.logger.info('Trained in {:.2f}s (n={}).'.format(time.time() - start, n))

    def predict(self, data, target, per=None):
        n = data.n

        if per is 'original':
            per = self.predict_per

        start = time.time()
        x_pred = self.representation.compute(data)
        if self.print_timings:
            cml.logger.info('Computed prediction representation in {:.2f}s (n={}).'.format(time.time() - start, n))

        start = time.time()

        y_pred = self.regressor.predict(x_pred)

        if self.print_timings:
            cml.logger.info('Predicted in {:.2f}s (n={}).'.format(time.time() - start, n))

        if per is None:
            return y_pred
        else:
            y_pred_per_atom = cml.unconvert(data, y_pred, self.predict_per)
            return cml.convert(data, y_pred_per_atom, per)

        return y_pred

    def loss(self, data, target, per=None, lossf='rmse'):
        lossf = cml.get_loss(lossf)

        if per is 'original':
            per = self.predict_per

        if isinstance(lossf, list):
            return self.losses(data, target, per=per, lossfs=lossf)

        pred = self.predict(data, target, per=per)
        true = data.pp(target, per=per)
        if lossf.needs_pv:
            raise NotImplementedError("Predictive variance is not yet implemented!")

        return lossf(true, pred, pv=None)

    def losses(self, data, target, per=None, lossfs=['rmse', 'mae', 'r2']):
        lossfs = cml.get_loss(lossfs)

        if per is 'original':
            per = self.predict_per

        true = data.pp(target, per=per)
        pred = self.predict(data, target, per=per)

        if any(l.needs_pv for l in lossfs):
            raise NotImplementedError("Predictive variance is not yet implemented!")

        result = {lossf.name: lossf(true, pred, pv=None) for lossf in lossfs}

        return result

    def cv_losses(self, data, idx, target, per=None, lossfs=['rmse', 'mae', 'r2']):
        """Compute cross-validation loss over data with idx.

        idx is expected to be a list of lists [idx_train, idx_test], where both
        are index arrays (i.e. int arrays) or something that can be cast to it.
        """
        start_all = time.time()

        lossfs = cml.get_loss(lossfs)

        if per is 'original':
            per = self.predict_per

        start = time.time()
        x = self.representation.compute(data)
        if self.print_timings:
            cml.logger.info(f"Computed CV representation in {time.time()-start:.2f}")

        y = data.pp(target, per=self.predict_per)  # converted to quantity internally used by regressor

        if hasattr(self.regressor, 'cv_train_and_predict') and not self.use_naive_cv:
            preds = self.regressor.cv_train_and_predict(x, y, idx)
        else:
            preds = self._naive_cv(x, y, data, idx, per)

        y_test = data.pp(target, per=per)  # converted to the quantity we want to predict

        result = {lossf.name: {'losses': []} for lossf in lossfs}
        for i, pred in enumerate(preds):
            test = idx[i][1]  # we only need the test indices

            if per is not None:
                pred = cml.unconvert(data[test], pred, self.predict_per)
                pred = cml.convert(data[test], pred, per)

            if any(l.needs_pv for l in lossfs):
                raise NotImplementedError("Predictive variance is not yet implemented!")

            for lossf in lossfs:
                result[lossf.name]['losses'].append(lossf(y_test[test], pred, pv=None))

        for lossf in lossfs:
            result[lossf.name]['losses'] = np.array(result[lossf.name]['losses'])
            result[lossf.name]['mean'] = np.mean(result[lossf.name]['losses'])
            result[lossf.name]['std'] = np.std(result[lossf.name]['losses'])
            result[lossf.name]['var'] = np.var(result[lossf.name]['losses'])


        final = {**{lossf.name: result[lossf.name]['mean'] for lossf in lossfs}, 'cv': result}

        if self.print_timings:
            cml.logger.info(f"Finished cv_losses in {time.time()-start_all:.2f}.")

        return final

    def _naive_cv(self, x, y, data, idx, per):
        # If the regressor does not offer an optimised CV routine,
        # this is used. (Might also be preferable if only small subsets
        # of the data are used in CV, since the regressor CV routines
        # should be optimised around the assumption that all data gets
        # used during the cross-validation.)

        start = time.time()
        preds = []
        for train, test in idx:

            self.regressor.train(x[train], y[train])
            pred = self.regressor.predict(x[test])

            if per is not None:
                pred = cml.unconvert(data[test], pred, self.predict_per)
                pred = cml.convert(data[test], pred, per)

            preds.append(pred)

        if self.print_timings:
            cml.logger.info(f"Finished naive CV in {time.time()-start:.2f}")

        return preds
