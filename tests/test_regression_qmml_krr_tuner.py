from unittest import TestCase
import numpy as np
import pathlib
import shutil

from cmlkit.regression.qmml.krr_tuner import KRRTunerCV
from cmlkit.regression.qmml import KRR
from cmlkit.utility.indices import twoway_split
from cmlkit import from_config


def f(x):
    return x.flatten() ** 3


def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))


class TestKRRTuner(TestCase):
    def setUp(self):
        self.x_train = 16 * np.random.random((80, 1)) - 8
        self.x_test = 16 * np.random.random((20, 1)) - 8

        self.y_train = f(self.x_train)
        self.y_test = f(self.x_test)

        self.idx = [twoway_split(80, 20) for i in range(3)]

        self.tuner = KRRTunerCV(
            kind_kernel="kernel_global",
            kind_kernelf="gaussian",
            optimizer={"opt_lgs": {"resolution": None, "rng": 123, "maxevals": 25}},
        )
        self.tuner.prepare(x=self.x_train, y=self.y_train, idx=self.idx)

        baseline = KRR(
            kernel={"kernel_global": {"kernelf": {"gaussian": {"ls": 8.0}}}}, nl=1.0
        )
        baseline.train(x=self.x_train, y=self.y_train)
        pred = baseline.predict(z=self.x_test)

        self.baseline_loss = rmse(self.y_test, pred)

    def test_does_it_improve(self):
        config, result = self.tuner.tune((0.0,), (3.0,))

        krr = from_config(config)
        krr.train(x=self.x_train, y=self.y_train)
        pred = krr.predict(z=self.x_test)

        loss = rmse(self.y_test, pred)

        self.assertLess(loss, self.baseline_loss)
