from unittest import TestCase
import os
from typing import List, Set
import pandas as pd
from ..etl import Extractor


class ExtractorTestCase(TestCase):

    def setUp(self):
        self.testpath: str = os.path.dirname(os.path.abspath(__file__))
        self.datadir = os.path.abspath(os.path.join(self.testpath, '..', 'data'))
        self.train_datadir = os.path.join(self.datadir, "cs-train")

        # things
        self.train_df_columns: List[str] = ['country', 'customer_id', 'day', 'invoice', 'month',
                                            'price', 'stream_id', 'times_viewed', 'year', 'invoice_date']
        self.train_df_column_num: int = 10
        self.train_df_row_num: int = 815011
        self.train_countries: Set[str] = {'United Kingdom', 'France', 'USA', 'Belgium', 'Australia', 'EIRE', 'Germany',
                                          'Portugal', 'Japan', 'Denmark', 'Nigeria', 'Netherlands', 'Poland', 'Spain',
                                          'Channel Islands', 'Italy', 'Cyprus', 'Greece', 'Norway', 'Austria', 'Sweden',
                                          'United Arab Emirates', 'Finland', 'Switzerland', 'Unspecified', 'Malta',
                                          'Bahrain', 'RSA', 'Bermuda', 'Hong Kong', 'Singapore', 'Thailand', 'Israel',
                                          'Lithuania', 'West Indies', 'Lebanon', 'Korea', 'Brazil', 'Canada', 'Iceland',
                                          'Saudi Arabia', 'Czech Republic', 'European Community'}
        self.train_total_revenue = 3914197.366
        self.train_beginning_date = pd.to_datetime("2017-11-28")
        self.train_end_date = pd.to_datetime("2019-07-31")

        # build up extractor
        self.extractor: Extractor = Extractor(self.train_datadir)
        self.df: pd.DataFrame = self.extractor.run()

    def tearDown(self):
        del self.df
        del self.extractor


class TestExtractorSingleton(ExtractorTestCase):

    def test_output_sizing(self):
        self.assertListEqual(self.df.columns.tolist(), self.train_df_columns)
        self.assertEqual(self.df.shape[1], self.train_df_column_num)
        self.assertEqual(self.df.shape[0], self.train_df_row_num)

    def test_output_values(self):
        self.assertSetEqual(set(self.df.country.unique()), self.train_countries)
        self.assertAlmostEqual(self.df['price'].sum(), self.train_total_revenue)
        self.assertAlmostEqual(self.df['invoice_date'].min(), self.train_beginning_date)
        self.assertAlmostEqual(self.df['invoice_date'].max(), self.train_end_date)
