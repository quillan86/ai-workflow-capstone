from unittest import TestCase
import os
import numpy as np
from typing import List, Dict
from src.app.pipeline.model import ModelContainer

class ModelContainerTestCase(TestCase):

    def setUp(self):
        self.datadir: str = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'data'))
        self.log: bool = False
        self.filename: str = "initial_model.db"

        self.country: str = "all_countries"


        self.model_scores = {
            'all_countries': 21912.566076053798,
            'united kingdom': 22942.383817131085,
            'eire': 1477.0867359247327,
            'germany': 424.0242650702814,
            'france': 437.3837510053454,
            'norway': 191.79684259795127,
            'spain': 272.466102683205,
            'hong kong': 755.9360424420678,
            'portugal': 412.9837558692272,
            'singapore': 2060.404732578652
        }

        self.y_true_date: float = 276604.0162401151
        self.y_true_range: np.ndarray = np.array([185003.50109837, 183755.54409948, 183970.28548068, 184524.86866884,
                                          177888.88098308, 177733.19507734, 172579.60871821, 172579.60871821,
                                          158329.18373278, 181823.90171696], dtype=float)

        self.model_container = ModelContainer(self.datadir, log=self.log)
        # load data
        self.model_container.load(self.filename)

    def tearDown(self):
        pass


class TestModelContainerSingleton(ModelContainerTestCase):

    def test_score(self):
        scores: Dict[str, float] = self.model_container.score()
        for key in self.model_scores:
            self.assertAlmostEqual(self.model_scores[key], scores[key])

    def test_predict_date(self):
        date = "2019-10-10"
        y_pred = self.model_container.predict_date(self.country, date)
        y_pred = float(y_pred[0])

        self.assertAlmostEqual(self.y_true_date, y_pred)

    def test_predict_range(self):
        initial_date: str = "2019-09-01"
        final_date: str = "2019-09-10"
        y_pred: np.ndarray = self.model_container.predict_range(self.country, initial_date, final_date)

        np.testing.assert_almost_equal(self.y_true_range, y_pred)

