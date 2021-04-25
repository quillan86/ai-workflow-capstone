import os
import shutil
from typing import List, Optional
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

    def save(self, train_only: bool = False, N: int = 15) -> pd.DataFrame:

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
        table = pd.pivot_table(df ,index='country', values="price", aggfunc='sum')
        table.columns = ['total_revenue']
        table.sort_values(by='total_revenue' ,inplace=True ,ascending=False)
        top_countries =  np.array(list(table.index))[:N].tolist()
        # iterative over top N countries
        for country in top_countries + [None]:
            transform_loader: TransformLoader = TransformLoader(country=country, revise_type=True, train_only=train_only)
            df_time = transform_loader.run(df)
            if country is not None:
                ts_filename = os.path.join(ts_datadir, f"{country.lower()}.csv")
            else:
                ts_filename = os.path.join(ts_datadir, f"all_countries.csv")
            df_time.to_csv(ts_filename, index=False)

        return df

    def load(self, country: Optional[str] = None, datatype: str = 'all', droptype: bool = False) -> pd.DataFrame:
        """
        datatypes: all, train, or prod in string format
        """
        # avoided the workaround that this is a string.
        if country is not None:
            if country.lower() == 'all_countries':
                country = None

        if country:
            c = country.lower()
            onlyfiles: List[str] = [f for f in os.listdir(self.ts_datadir) if os.path.isfile(os.path.join(self.ts_datadir, f))]
            countries_in_dir: List[str] = [c.split('.')[0] for c in onlyfiles]
            if c not in countries_in_dir:
                raise Exception("country not found")
            all_countries: bool = False
        else:
            # all countries if None
            all_countries: bool = True

        if all_countries:
            filename: str = os.path.join(self.ts_datadir, 'all_countries.csv')
        else:
            filename: str = os.path.join(self.ts_datadir, f'{c}.csv')
        df_ts: pd.DataFrame = pd.read_csv(filename, parse_dates=['date'])
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

    def _add_country(self, filename: str, country: str):
        df = pd.read_csv(filename)
        df[country] = country