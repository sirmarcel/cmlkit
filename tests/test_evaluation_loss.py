from unittest import TestCase
import numpy as np

from cmlkit.evaluation.loss.lossfs import *
from cmlkit.evaluation.loss.loss import Loss, get_loss


class TestLossfs(TestCase):
    def test_vs_qmmlpack(self):
        from qmmlpack import loss
        for i in range(5):
            true = np.random.random(100)
            pred = np.random.random(100)

            self.assertEqual(rmse(true, pred), loss(true, pred, lossf="rmse"))
            self.assertEqual(mae(true, pred), loss(true, pred, lossf="mae"))
            self.assertEqual(maxae(true, pred), loss(true, pred, lossf="maxae"))
            self.assertEqual(medianae(true, pred), loss(true, pred, lossf="medianae"))
            np.testing.assert_almost_equal(r2(true, pred), loss(true, pred, lossf="r2"))

    def test_get_lossf(self):
        self.assertEqual(rmse, get_lossf("rmse"))
        self.assertEqual(rmse, get_lossf(rmse))

        with self.assertRaises(ValueError):
            get_lossf("non existing")


class TestLoss(TestCase):
    def test_basic(self):
        loss = Loss("rmse")

        true = np.random.random(100)
        pred = np.random.random(100)

        self.assertEqual(rmse(true, pred), loss(true, pred)["rmse"])

    def test_shortcut(self):
        loss = get_loss("default")

        self.assertEqual(loss.lossfs, [rmse, mae, r2])

    def test_spec(self):
        loss = get_loss("rmse", "mae")
        spec = loss.spec
        loss = get_loss(spec)

        self.assertEqual(loss.lossfs, [rmse, mae])
        self.assertEqual(loss.spec, ["rmse", "mae"])

    def test_magic_syntax(self):
        loss = get_loss(["rmse", "mae"])
        loss2 = get_loss("rmse", "mae")

        self.assertEqual(loss.spec, loss2.spec)

    def test_needs_pv(self):
        loss = get_loss("rmse", "mnlp")
        self.assertTrue(loss.needs_pv)

        loss = get_loss("rmse", "mae")
        self.assertFalse(loss.needs_pv)
