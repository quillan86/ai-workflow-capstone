from unittest import TestCase
from fastapi.testclient import TestClient
import numpy as np
from src.app.main import app
from typing import List


class APITestCase(TestCase):

    def setUp(self):
        self.client = TestClient(app)

        self.score_result = {
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

        self.predict_date_result = {
                                     "country": "all_countries",
                                     "initial_date": "2019-10-10",
                                     "forecasted_date": "2019-11-09",
                                     "forecasted_revenue": 276604.0162401151
                                     }

        initial_dates: List[str] = [
            "2019-09-01",
            "2019-09-02",
            "2019-09-03",
            "2019-09-04",
            "2019-09-05",
            "2019-09-06",
            "2019-09-07",
            "2019-09-08",
            "2019-09-09",
            "2019-09-10"
        ]

        forecasted_dates: List[str] = [
        "2019-10-01",
        "2019-10-02",
        "2019-10-03",
        "2019-10-04",
        "2019-10-05",
        "2019-10-06",
        "2019-10-07",
        "2019-10-08",
        "2019-10-09",
        "2019-10-10"
      ]

        forecasted_revenue: List[float] = [
        185003.5010983733,
        183755.54409948495,
        183970.28548067648,
        184524.8686688435,
        177888.88098307967,
        177733.19507734315,
        172579.60871821395,
        172579.60871821395,
        158329.183732778,
        181823.90171695876
      ]

        self.forecasted_range_results = {
                                     "country": "all_countries",
                                     "initial_dates": initial_dates,
                                     "forecasted_dates": forecasted_dates,
                                     "forecasted_revenue": forecasted_revenue
                                     }

    def tearDown(self):
        del self.client


class TestAPISingleton(APITestCase):

    def test_scores(self):
        model = "initial_model"
        response = self.client.get(f"api/v1/model/train/?name={model}")
        self.assertEquals(response.status_code, 200)
        result = response.json()

        for key in self.score_result.keys():
            self.assertAlmostEqual(self.score_result[key], result[key])

    def test_predict_date(self):
        model: str = "initial_model"
        country: str = "all_countries"
        date: str = "2019-10-10"
        response = self.client.get(f"api/v1/model/forecast_date/?name={model}&country={country}&date={date}")
        self.assertEquals(response.status_code, 200)
        result = response.json()

        for key in self.predict_date_result.keys():
            if key == "forecasted_revenue":
                self.assertAlmostEqual(self.predict_date_result[key], result[key])
            else:
                self.assertEqual(self.predict_date_result[key], result[key])

    def test_predict_range(self):
        model: str = "initial_model"
        country: str = "all_countries"
        initial_date: str = "2019-09-01"
        final_date: str = "2019-09-10"

        response = self.client.get(f"api/v1/model/forecast_range/?name={model}&country={country}&initial_date={initial_date}&final_date={final_date}")

        self.assertEquals(response.status_code, 200)
        result = response.json()

        for key in self.forecasted_range_results.keys():
            if key == "forecasted_revenue":
                np.testing.assert_almost_equal(self.forecasted_range_results[key], result[key])
            elif key in ["initial_dates", "forecasted_dates"]:
                for item1, item2 in zip(self.forecasted_range_results[key], result[key]):
                    self.assertEqual(item1, item2)
            else:
                self.assertEqual(self.forecasted_range_results[key], result[key])
