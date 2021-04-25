from unittest import TestCase
import os
import pandas as pd

from ...pipeline.logging import Logger


class LoggerTestCase(TestCase):

    def setUp(self):
        self.logger = Logger()
        self.logdir = self.logger.logdir

    def tearDown(self):
        del self.logger


class TestLoggerSingleton(LoggerTestCase):

    def test_01_train(self):
        """
        ensure log file is created
        """

        log_file = os.path.join(self.logdir, "train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        data_shape = (100, 10)
        eval_test = 0.5
        runtime = "00:00:01"
        model_version = "0.1"
        country = 'all_countries'
        model_version_note = "test model"

        self.logger.update_train_log(data_shape, eval_test, runtime, country,
                                     model_version, model_version_note, test=True)

        self.assertTrue(os.path.exists(log_file))

    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join(self.logdir, "train-test.log")

        ## update the log
        data_shape = (100, 10)
        eval_test = 0.5
        runtime = "00:00:01"
        model_version = "0.1"
        country = 'all_countries'
        model_version_note = "test model"

        self.logger.update_train_log(data_shape, eval_test, runtime, country,
                         model_version, model_version_note, test=True)

        df = pd.read_csv(log_file)
        logged_eval_test = df.iloc[-1, 3]
        self.assertEqual(eval_test, logged_eval_test)

    def test_03_predict(self):
        """
        ensure log file is created
        """

        log_file = os.path.join(self.logdir, "predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        y_pred = 183755.54409948495
        y_proba = 'None'
        runtime = "00:00:02"
        model_version = "0.1"
        country = "all_countries"
        query = [9664.0, 17741.0, 32444.0, 85129.0, 44497.0, 408.0, 838.0, 1584.0, 4197.0, 2391.0, 4899.0, 9581.0, 18413.0, 49264.0, 22737.0, 43447.0, 95708.0, 186720.0, 471964.0, 265188.0, 39854.02, 77294.03, 130274.95999999999, 360699.26100000006, 229918.98099999997]

        self.logger.update_predict_log(y_pred, y_proba, query, runtime, country,
                           model_version, test=True)

        self.assertTrue(os.path.exists(log_file))

    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join(self.logdir, "predict-test.log")

        ## update the log
        y_pred = 183755.54409948495
        y_proba = 'None'
        runtime = "00:00:02"
        model_version = "0.1"
        country = "all_countries"
        query = [9664.0, 17741.0, 32444.0, 85129.0, 44497.0, 408.0, 838.0, 1584.0, 4197.0, 2391.0, 4899.0, 9581.0,
                 18413.0, 49264.0, 22737.0, 43447.0, 95708.0, 186720.0, 471964.0, 265188.0, 39854.02, 77294.03,
                 130274.95999999999, 360699.26100000006, 229918.98099999997]

        self.logger.update_predict_log(y_pred, y_proba, query, runtime, country,
                           model_version, test=True)

        df = pd.read_csv(log_file)
        logged_y_pred = df.iloc[-1, 2]
        self.assertAlmostEqual(y_pred, logged_y_pred)

