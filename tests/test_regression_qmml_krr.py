from unittest import TestCase
import numpy as np
import pathlib
import shutil

from cmlkit.regression.qmml import KRR


def f(x):
    return x.flatten() ** 3


def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))


class TestKRR(TestCase):
    def setUp(self):
        self.x_train = 4 * np.random.random((160, 1)) - 2
        self.x_test = 4 * np.random.random((40, 1)) - 2

        self.y_train = f(self.x_train)
        self.y_test = f(self.x_test)

        # self.tmpdir = (pathlib.Path(__file__) / "..").resolve() / "tmp_test_krr"
        # self.tmpdir.mkdir(exist_ok=True)


    # def tearDown(self):
    #     shutil.rmtree(self.tmpdir)


    def test_does_it_work(self):
        krr = KRR(
            kernel={"qmml_kernel_global": {"kernelf": {"gaussian": {"ls": 0.5}}}},
            nl=1.0e-7,
        )

        krr.train(x=self.x_train, y=self.y_train)

        p = krr.predict(self.x_test)

        loss = rmse(self.y_test, p)

        # uncomment this for visual debugging ;-)

        # import matplotlib
        # matplotlib.use('tkagg')
        # import matplotlib.pyplot as plt
        # x = np.linspace(-2, 2, num=100)
        # plt.plot(x, f(x))
        # plt.scatter(self.x_test.flatten(), p)
        # plt.savefig(self.tmpdir / "lol.pdf")

        self.assertLess(loss, 0.001)
