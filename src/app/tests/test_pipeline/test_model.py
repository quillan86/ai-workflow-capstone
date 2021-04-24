from unittest import TestCase
import os
import numpy as np
from typing import List
from src.app.pipeline.model import Model

class ModelTestCase(TestCase):

    def setUp(self):
        self.datadir: str = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'data'))
        self.log: bool = False
        self.seed: int = 42
        self.country: str = 'united kingdom'

        self.score_mean: float = 22947.16597656205
        self.score_std: float = 2936.910358408868

        self.y_pred_beg: float = 157709.0533772044
        self.y_pred_end: float = 162114.28099755727

        self.model: Model = Model(self.datadir, self.country, log=self.log, seed=self.seed)

    def tearDown(self):
        del self.model

class TestModelSingleton(ModelTestCase):

    def test_training(self):
        X_train, y_train, dates_train = self.model.load_train_data()
        self.model.fit(X_train, y_train)
        score_mean, score_std = self.model.score(X_train, y_train)

        y_pred: np.ndarray = self.model.predict(X_train.iloc[[0, -1], :])

        self.assertAlmostEqual(score_mean, self.score_mean)
        self.assertAlmostEqual(score_std, self.score_std)
        self.assertAlmostEqual(y_pred[0], self.y_pred_beg)
        self.assertAlmostEqual(y_pred[1], self.y_pred_end)
