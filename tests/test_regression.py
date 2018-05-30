from unittest import TestCase
import numpy as np
from qmmltools.regression import *
from qmmltools.dataset import read
from qmmltools.model_spec import ModelSpec
from qmmltools.mbtr.mbtr import MBTR
import os
dirname = os.path.dirname(os.path.abspath(__file__))

data = read(dirname + '/kaggle-mini.dat.npy')
spec = ModelSpec.from_yaml(dirname + '/model_mini.spec.yml')
rep = MBTR(data, spec)
train = np.arange(10)
predict = np.arange(11, 20)


class TestKernel(TestCase):

    def test_works_for_all_kernels(self):
        compute_kernel({'kernelf': ('gaussian', 1.0)}, rep.raw)
        compute_kernel({'kernelf': ('laplacian', 1.0)}, rep.raw)
        compute_kernel({'kernelf': ('linear')}, rep.raw)
        # let's just make sure this doesn't break


class TestTrainModel(TestCase):

    def test_works_with_given_kernelm(self):
        kernel_matrix = compute_kernel(spec.krr, rep.raw)
        train_model(data, spec, kernel_matrix)
        # will produce an error if something is broken

    def test_works_without_given_kernelm_but_with_rep(self):
        train_model(data, spec, rep=rep)
        # will produce an error if something is broken

    def test_works_without_given_kernelm(self):
        train_model(data, spec)
        # will produce an error if something is broken


class TestIDXTrainAndPredict(TestCase):

    def test_works_with_same_property(self):
        idx_train_and_predict(data, spec, train, predict, rep=rep)
        # will throw error if something is broken

    def test_works_with_same_property_and_missing_rep(self):
        idx_train_and_predict(data, spec, train, predict)
        # will throw error if something is broken

    def test_works_with_different_property(self):
        idx_train_and_predict(data, spec, train, predict, target_property='fe', rep=rep)
        # will throw error if something is broken


class TestIDXLoss(TestCase):

    def test_works_with_same_property(self):
        idx_compute_loss(data, spec, train, predict, rep=rep)
        # will throw error if something is broken

    def test_works_with_different_property(self):
        idx_compute_loss(data, spec, train, predict, target_property='fe', rep=rep)
        # will throw error if something is broken
