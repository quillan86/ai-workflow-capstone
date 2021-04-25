import os
import pandas as pd
from unittest import TestCase

from src.app.pipeline.state import State

class StateTestCase(TestCase):

    def setUp(self):
        self.datadir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'data'))
        self.train_datadir = os.path.join(self.datadir, "cs-train")
        self.prod_datadir = os.path.join(self.datadir, "cs-production")
        self.ts_datadir: str = os.path.join(self.datadir, 'ts-data')

        self.train_beginning_date = pd.to_datetime("2017-11-01")
        self.train_end_date = pd.to_datetime("2019-07-31")

        self.state = State(self.datadir)

    def tearDown(self):
        pass


class TestStateSingleton(StateTestCase):

    def test_save(self):
        # save data
        N: int = 15
        self.state.save(train_only=False, N=N)

        # look at data structure
        files = [os.path.join(self.ts_datadir, f) for f in os.listdir(self.ts_datadir) if os.path.isfile(os.path.join(self.ts_datadir, f))]

        self.assertEqual(len(files), N + 1) # top 15 + all countries

    def test_load_uk(self):
        # load data
        df_uk = self.state.load(country='United Kingdom')

        train_uk: pd.DataFrame = df_uk[df_uk['type'] == 'train']
        prod_uk: pd.DataFrame = df_uk[df_uk['type'] == 'prod']

        purchase_sum: int = train_uk['purchases'].sum()
        unique_invoice_sum: int = train_uk['unique_invoices'].sum()
        unique_stream_sum: int = train_uk['unique_streams'].sum()
        total_view_sum: int = train_uk['total_views'].sum()
        revenue_sum: float = train_uk['revenue'].sum()

        self.assertEqual(len(train_uk), 638)
        self.assertEqual(len(df_uk)-len(train_uk), len(prod_uk))

        self.assertEqual(purchase_sum, 751228)
        self.assertEqual(unique_invoice_sum, 39256)
        self.assertEqual(unique_stream_sum, 402528)
        self.assertEqual(total_view_sum, 3725703)
        self.assertAlmostEqual(revenue_sum, 3521513.505)

        self.assertAlmostEqual(train_uk.loc[0, 'date'], self.train_beginning_date)
        self.assertAlmostEqual(train_uk.loc[637, 'date'], self.train_end_date)

    def test_load_all(self):
        # load data
        df_all = self.state.load(country=None)
        train_all: pd.DataFrame = df_all[df_all['type'] == 'train']
        prod_all: pd.DataFrame = df_all[df_all['type'] == 'prod']

        purchase_sum: int = train_all['purchases'].sum()
        unique_invoice_sum: int = train_all['unique_invoices'].sum()
        unique_stream_sum: int = train_all['unique_streams'].sum()
        total_view_sum: int = train_all['total_views'].sum()
        revenue_sum: float = train_all['revenue'].sum()

        self.assertEqual(len(train_all), 638)
        self.assertEqual(len(df_all)-len(train_all), len(prod_all))

        self.assertEqual(purchase_sum, 815011)
        self.assertEqual(unique_invoice_sum, 42646)
        self.assertEqual(unique_stream_sum, 423802)
        self.assertEqual(total_view_sum, 4263409)
        self.assertAlmostEqual(revenue_sum, 3914197.366)

        self.assertAlmostEqual(train_all.loc[0, 'date'], self.train_beginning_date)
        self.assertAlmostEqual(train_all.loc[637, 'date'], self.train_end_date)
