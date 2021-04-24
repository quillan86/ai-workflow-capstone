from unittest import TestCase
import os
import pandas as pd
from typing import List
from ..pipeline.features import FeatureEngineer
from ..pipeline.state import State

class FeatureEngineerTestCase(TestCase):

    def setUp(self):
        self.datadir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
        self.log: bool = False
        self.uk: str = 'united kingdom'

        self.uk_X_train_shape: tuple = (580, 25)
        self.uk_feature_columns: List[str] = ['previous_7_purchases', 'previous_14_purchases', 'previous_28_purchases', 'previous_70_purchases', 'previous_year_purchases',
                                              'previous_7_unique_invoices', 'previous_14_unique_invoices', 'previous_28_unique_invoices', 'previous_70_unique_invoices', 'previous_year_unique_invoices',
                                              'previous_7_unique_streams', 'previous_14_unique_streams', 'previous_28_unique_streams', 'previous_70_unique_streams', 'previous_year_unique_streams',
                                              'previous_7_total_views', 'previous_14_total_views', 'previous_28_total_views', 'previous_70_total_views', 'previous_year_total_views',
                                              'previous_7_revenue', 'previous_14_revenue', 'previous_28_revenue', 'previous_70_revenue', 'previous_year_revenue']
        self.train_beginning_date = pd.to_datetime("2017-11-29")
        self.train_end_date = pd.to_datetime("2019-07-01")

        self.state: State = State(self.datadir)
        self.df_uk: pd.DataFrame = self.state.load(country=self.uk, datatype ='all', droptype=False)
        self.feature_engineer: FeatureEngineer = FeatureEngineer(datatype='train', log=self.log)

    def tearDown(self):
        del self.state
        del self.feature_engineer


class TestFeatureEngineerSingleton(FeatureEngineerTestCase):

    def test_features(self):
        X_train, y_train, dates_train = self.feature_engineer.run(self.df_uk)

        self.assertEqual(X_train.columns.tolist(), self.uk_feature_columns)
        self.assertEqual(X_train.shape, self.uk_X_train_shape)
        self.assertEqual(len(y_train), self.uk_X_train_shape[0])
        self.assertEqual(len(dates_train), self.uk_X_train_shape[0])
        self.assertEqual(pd.to_datetime(dates_train[0]), self.train_beginning_date)
        self.assertEqual(pd.to_datetime(dates_train[-1]), self.train_end_date)
