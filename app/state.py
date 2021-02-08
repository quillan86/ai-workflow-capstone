import os
import shutil
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
import pandas as pd

from .etl import Extractor, TransformLoader


class State:
    def __init__(self, datadir):
        self.datadir = datadir
        self.train_datadir = os.path.join(self.datadir, "cs-train")
        self.prod_datadir = os.path.join(self.datadir, "cs-production")
        self.ts_datadir = os.path.join(self.datadir, "ts-data")

        self.train_fetcher = Extractor(self.train_datadir)
        self.prod_fetcher = Extractor(self.prod_datadir)

    @staticmethod
    def revise_ts(df_time: pd.DataFrame) -> pd.DataFrame:
        """
        Add type of data - train or production data. Training data is until 8/1/2019
        """
        df_time['type'] = ''
        df_time.loc[(((df_time['year'] == 2019) & (df_time['month'] >= 8)) | (df_time['year'] > 2019)), 'type'] = 'prod'
        df_time.loc[(df_time['type'] != 'prod'), 'type'] = 'train'
        return df_time

    def save(self) -> pd.DataFrame:

        ts_datadir: str = os.path.join(self.datadir, 'ts-data')

        # wipe old files
        shutil.rmtree(ts_datadir)
        if not os.path.exists(ts_datadir):
            os.mkdir(ts_datadir)

        df_train = self.train_fetcher.run()
        df_prod = self.prod_fetcher.run()

        df_train['type'] = 'train'
        df_prod['type'] = 'prod'
        df = pd.concat([df_train, df_prod])

        ## find the top N countries (wrt revenue)
        N: int = 15
        table = pd.pivot_table(df ,index='country', values="price", aggfunc='sum')
        table.columns = ['total_revenue']
        table.sort_values(by='total_revenue' ,inplace=True ,ascending=False)
        top_countries =  np.array(list(table.index))[:N].tolist()

        for country in top_countries:
            transform_loader: TransformLoader = TransformLoader(country=country)
            df_time = transform_loader.run(df)
            df_time = self.revise_ts(df_time)
            ts_filename = os.path.join(ts_datadir, f"{country.lower()}.csv")
            df_time.to_csv(ts_filename, index=False)

        return df

    def load(self, country: Optional[str] = None, datatype: str = 'all', droptype: bool = False) -> pd.DataFrame:
        """
        datatypes: all, train, or prod in string format
        """

        if country:
            c = country.lower()
            onlyfiles: List[str] = [f for f in os.listdir(self.ts_datadir) if os.path.isfile(os.path.join(self.ts_datadir, f))]
            countries_in_dir: List[str] = [c.split('.')[0] for c in onlyfiles]
            if c not in countries_in_dir:
                raise Exception("country not found")
        else:
            raise Exception("You must select a country.")

        filename: str = os.path.join(self.ts_datadir, f'{c}.csv')
        df_ts: pd.DataFrame = pd.read_csv(filename)
        if datatype.lower() == 'train':
            df_ts = df_ts[df_ts['type'] == 'train']
        elif datatype.lower() == 'prod':
            df_ts = df_ts[df_ts['type'] == 'prod']
        elif datatype.lower() == 'all':
            # all data
            pass
        else:
            # all data
            raise Exception("datatype must be train, prod, or all")

        # remove type column
        if droptype:
            df_ts = df_ts.drop(columns=['type'])
        return df_ts
