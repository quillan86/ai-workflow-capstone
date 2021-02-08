import os
import re
from typing import Optional, List, Dict, Set
import numpy as np
import pandas as pd


class Extractor:
    def __init__(self, datadir: str):
        """
        Fetch data from JSON.
        """
        self.datadir: str = datadir

    def run(self) -> pd.DataFrame:
        """
        laod all json formatted files into a dataframe
        """

        # input testing
        if not os.path.isdir(self.datadir):
            raise Exception("specified data dir does not exist")
        if not len(os.listdir(self.datadir)) > 0:
            raise Exception("specified data dir does not contain any files")

        file_list: List[str] = [os.path.join(self.datadir ,f) for f in os.listdir(self.datadir) if re.search("\.json" ,f)]
        correct_columns: List[str] = ['country', 'customer_id', 'day', 'invoice', 'month',
                                      'price', 'stream_id', 'times_viewed', 'year']

        # read data into a temp structure
        all_months: Dict[str, pd.DataFrame] = {}
        for file_name in file_list:
            df: pd.DataFrame = pd.read_json(file_name)
            all_months[os.path.split(file_name)[-1]] = df

        # ensure the data are formatted with correct columns
        for f ,df in all_months.items():
            cols: Set[str] = set(df.columns.tolist())
            if 'StreamID' in cols:
                df.rename(columns={'StreamID' :'stream_id'} ,inplace=True)
            if 'TimesViewed' in cols:
                df.rename(columns={'TimesViewed' :'times_viewed'} ,inplace=True)
            if 'total_price' in cols:
                df.rename(columns={'total_price' :'price'} ,inplace=True)

            cols: List[str] = df.columns.tolist()
            if sorted(cols) != correct_columns:
                raise Exception("columns name could not be matched to correct cols")

        # concat all of the data
        df: pd.DataFrame = pd.concat(list(all_months.values()) ,sort=True)
        years: np.ndarray = df['year'].values
        months: np.ndarray = df['month'].values
        days: np.ndarray =  df['day'].values


        dates: List[str] = ["{}-{}-{}".format(years[i] ,str(months[i]).zfill(2) ,str(days[i]).zfill(2)) for i in range(df.shape[0])]
        df['invoice_date'] = np.array(dates ,dtype='datetime64[D]')
        df['invoice'] = [re.sub("\D+" ,"" ,i) for i in df['invoice'].values]

        # sort by date and reset the index
        df.sort_values(by='invoice_date' ,inplace=True)
        df.reset_index(drop=True ,inplace=True)

        return df


class TransformLoader:
    def __init__(self, country: Optional[str] = None):
        self.country: Optional[str] = country

    def run(self, df_orig: pd.DataFrame) -> pd.DataFrame:
        """
        given the original DataFrame (fetch_data())
        return a numerically indexed time-series DataFrame
        by aggregating over each day
        """

        if self.country:
            if self.country not in np.unique(df_orig['country'].values):
                raise Exception("country not found")
            mask: pd.Series = df_orig['country'] == self.country
            df: pd.DataFrame = df_orig[mask]
        else:
            df: pd.DataFrame = df_orig

        ## use a date range to ensure all days are accounted for in the data
        start_month: str = f"{df['year'].values[0]}-{str(df['month'].values[0]).zfill(2)}"
        stop_month: str = f"{df['year'].values[-1]}-{str(df['month'].values[-1]).zfill(2)}"
        df_dates: np.ndarray = df['invoice_date'].values.astype('datetime64[D]')
        days: np.ndarray = np.arange(start_month, stop_month, dtype='datetime64[D]')

        purchases: np.ndarray = np.array([np.where(df_dates == day)[0].size for day in days])
        invoices: List[int] = [np.unique(df[df_dates == day]['invoice'].values).size for day in days]
        streams: List[int] = [np.unique(df[df_dates == day]['stream_id'].values).size for day in days]
        views: List[int] =  [df[df_dates == day]['times_viewed'].values.sum() for day in days]
        revenue: List[float] = [df[df_dates == day]['price'].values.sum() for day in days]
        year_month: List[str] = ["-".join(re.split("-" ,str(day))[:2]) for day in days]

        df_time: pd.DataFrame = pd.DataFrame({'date' :days,
                                              'purchases' :purchases,
                                              'unique_invoices' :invoices,
                                              'unique_streams' :streams,
                                              'total_views' :views,
                                              'year_month' :year_month,
                                              'revenue' :revenue})

        # new - month and year
        df_time['month'] = df_time['year_month'].str.split('-').apply(lambda x: x[1]).astype(int)
        df_time['year'] =  df_time['year_month'].str.split('-').apply(lambda x: x[0]).astype(int)

        return df_time
