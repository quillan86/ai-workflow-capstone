from unittest import TestCase
from typing import Dict
from ...pipeline.monitoring import Monitor


class MonitorTestCase(TestCase):

    def setUp(self):

        self.filename: str = "initial_model.db"
        self.seed: int = 42

        self.monitor = Monitor(self.filename, seed=self.seed, log=False)

        self.monitor_results = {
            'portugal': {'outlier_X': 1.51, 'wasserstein_X': 27.02, 'wasserstein_y': 175.28},
            'norway': {'outlier_X': 0.86, 'wasserstein_X': 128.33, 'wasserstein_y': 66.64},
            'united kingdom': {'outlier_X': 2.16, 'wasserstein_X': 3095.55, 'wasserstein_y': 11166.89},
            'all_countries': {'outlier_X': 2.37, 'wasserstein_X': 3404.85, 'wasserstein_y': 11749.13},
            'singapore': {'outlier_X': 4.69, 'wasserstein_X': 74.01, 'wasserstein_y': 635.11},
            'spain': {'outlier_X': 2.16, 'wasserstein_X': 24.43, 'wasserstein_y': 127.44},
            'germany': {'outlier_X': 2.16, 'wasserstein_X': 61.17, 'wasserstein_y': 123.51},
            'france': {'outlier_X': 2.37, 'wasserstein_X': 56.98, 'wasserstein_y': 139.92},
            'hong kong': {'outlier_X': 4.38, 'wasserstein_X': 33.45, 'wasserstein_y': 274.3},
            'eire': {'outlier_X': 2.16, 'wasserstein_X': 117.08, 'wasserstein_y': 592.04}
        }


    def tearDown(self):
        pass


class TestMonitorSingleton(MonitorTestCase):

    def test_monitor(self):

        results: Dict[str, dict] = self.monitor.detect_all()

        for country in self.monitor_results.keys():
            monitor_result = self.monitor_results[country]
            result = results[country]
            for key in monitor_result:
                self.assertAlmostEqual(monitor_result[key], result[key])

