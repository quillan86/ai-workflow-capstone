import os
import pandas as pd
from unittest import TestCase

from ..pipeline.etl import Extractor, TransformLoader

class TransformLoaderTestCase(TestCase):

    def setUp(self):
        self.datadir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
        self.train_datadir = os.path.join(self.datadir, "cs-train")
        self.prod_datadir = os.path.join(self.datadir, "cs-production")
        self.ts_datadir: str = os.path.join(self.datadir, 'ts-data')

        self.train_fetcher = Extractor(self.train_datadir)
        self.prod_fetcher = Extractor(self.prod_datadir)

        self.df = self._load_json_data()

        self.uk_lowercase: str = 'united kingdom'
        self.uk_uppercase: str = 'United Kingdom'

        self.transformed_columns = ['date', 'purchases', 'unique_invoices', 'unique_streams', 'total_views',
                                    'year_month', 'revenue', 'month', 'year', 'type']

        self.train_beginning_date = pd.to_datetime("2017-11-01")
        self.train_end_date = pd.to_datetime("2019-07-31")

        # build up transform loader
        self.transform_loader_uk = TransformLoader(country=self.uk_lowercase, revise_type=True, train_only=True)
        self.transform_loader_all = TransformLoader(country=None, revise_type=True, train_only=False)

    def _load_json_data(self):
        """
        Load JSON data and place it in a dataframe, from both training and production
        """

        df_train = self.train_fetcher.run()
        df_prod = self.prod_fetcher.run()

        df_train['type'] = 'train'
        df_prod['type'] = 'prod'
        df = pd.concat([df_train, df_prod])

        return df

    def tearDown(self):
        del self.transform_loader_all
        del self.transform_loader_uk
        del self.df


class TestTransformLoaderSingleton(TransformLoaderTestCase):

    def test_country_data(self):
        df_uk_transformed: pd.DataFrame = self.transform_loader_uk.run(self.df)
        train_uk: pd.DataFrame = df_uk_transformed[df_uk_transformed['type'] == 'train']
        prod_uk: pd.DataFrame = df_uk_transformed[df_uk_transformed['type'] == 'prod']

        purchase_sum: int = train_uk['purchases'].sum()
        unique_invoice_sum: int = train_uk['unique_invoices'].sum()
        unique_stream_sum: int = train_uk['unique_streams'].sum()
        total_view_sum: int = train_uk['total_views'].sum()
        revenue_sum: float = train_uk['revenue'].sum()

        self.assertEqual(self.transform_loader_uk.country, self.uk_uppercase)
        self.assertEqual(df_uk_transformed.columns.tolist(), self.transformed_columns)

        self.assertEqual(len(train_uk), 638)
        self.assertEqual(len(prod_uk), 0) # since we set the transformed data is training only
        self.assertEqual(len(train_uk), len(df_uk_transformed))

        self.assertEqual(purchase_sum, 751228)
        self.assertEqual(unique_invoice_sum, 39256)
        self.assertEqual(unique_stream_sum, 402528)
        self.assertEqual(total_view_sum, 3725703)
        self.assertAlmostEqual(revenue_sum, 3521513.505)

        self.assertAlmostEqual(train_uk.loc[0, 'date'], self.train_beginning_date)
        self.assertAlmostEqual(train_uk.loc[637, 'date'], self.train_end_date)

    def test_all_data(self):
        df_all_transformed: pd.DataFrame = self.transform_loader_all.run(self.df)
        train_all: pd.DataFrame = df_all_transformed[df_all_transformed['type'] == 'train']
        prod_all: pd.DataFrame = df_all_transformed[df_all_transformed['type'] == 'prod']

        purchase_sum: int = train_all['purchases'].sum()
        unique_invoice_sum: int = train_all['unique_invoices'].sum()
        unique_stream_sum: int = train_all['unique_streams'].sum()
        total_view_sum: int = train_all['total_views'].sum()
        revenue_sum: float = train_all['revenue'].sum()

        self.assertEqual(self.transform_loader_all.country, None)
        self.assertEqual(df_all_transformed.columns.tolist(), self.transformed_columns)

        self.assertEqual(len(train_all), 638)
        self.assertEqual(len(df_all_transformed)-len(train_all), len(prod_all))

        self.assertEqual(purchase_sum, 815011)
        self.assertEqual(unique_invoice_sum, 42646)
        self.assertEqual(unique_stream_sum, 423802)
        self.assertEqual(total_view_sum, 4263409)
        self.assertAlmostEqual(revenue_sum, 3914197.366)

        self.assertAlmostEqual(train_all.loc[0, 'date'], self.train_beginning_date)
        self.assertAlmostEqual(train_all.loc[637, 'date'], self.train_end_date)
