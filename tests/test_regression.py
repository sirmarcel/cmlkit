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

    def test_works(self):
        kernel(spec, rep.raw)
        # let's just make sure this doesn't break


class TestTrainModel(TestCase):

    def test_works(self):
        kernel_matrix = kernel(spec, rep.raw)
        train_model(data, spec, kernel_matrix)
        # will produce an error if something is broken

class TestTrainAndPredict(TestCase):

    def test_works_with_same_property(self):
        train_and_predict(data, spec, rep, train, predict)
        # will throw error if something is broken

    def test_works_with_different_property(self):
        train_and_predict(data, spec, rep, train, predict, target_property='fe')
        # will throw error if something is broken

class TestLoss(TestCase):

    def test_works_with_same_property(self):
        loss(data, spec, rep, train, predict)
            # will throw error if something is broken

    def test_works_with_different_property(self):
        loss(data, spec, rep, train, predict, target_property='fe')
        # will throw error if something is broken
