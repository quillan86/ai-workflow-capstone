from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, datatype: str = 'all', log: bool = False, more_features: bool = False):
        """
        df: time series dataframe to transform into predictive features.
        datatype: one of (all, train, prod) to determine type of data to engineer over.
        more features - whether we will get more features for looking in the past.
        """
        self.datatype: str = datatype.lower()
        self.log: bool = log
        self.more_features: bool = more_features
        if self.datatype not in [None, 'all', 'train', 'prod']:
            raise Exception("datatype must be train, prod, or all")

    def refactor_original(self, df: pd.DataFrame) -> pd.DataFrame:

        columns_p1: List[str] = ["date", "purchases", "unique_invoices", "unique_streams", "total_views", "year_month"]
        columns_p2: List[str] = ["type", "revenue"]
        df: pd.DataFrame = pd.concat([df[columns_p1], pd.get_dummies(df.month, prefix='month'),
                                      pd.get_dummies(df.year, prefix='year'), df[columns_p2]], axis=1)

        # add months as new variables if not in variables
        for i in range(1, 13):
            month: str = f'month_{i:02d}'
            if month not in df.columns:
                df[month] = 0
                df[month] = df[month].astype(int)

        df = df[['date', 'purchases', 'unique_invoices', 'unique_streams', 'total_views',
                 'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                 'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                 'type', 'revenue']]
        return df

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        df: country based time series
        datatypes: all, train, or prod in string format
        """

        if self.datatype.lower() not in ['train', 'prod', 'all']:
            raise Exception("datatype must be train, prod, or all")

        # refactor
        df = self.refactor_original(df)

        ## extract dates
        dates: np.ndarray = df['date'].values.copy()
        dates: np.ndarray = dates.astype('datetime64[D]')

        ## engineer some features
        eng_features: defaultdict = defaultdict(list)

        if self.more_features:
            previous: List[int] = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
        else:
            previous: List[int] = [7, 14, 28, 70]
        y: np.ndarray = np.zeros(dates.size)
        # first is date, last is revenue
        variables: List[str] = ['purchases', 'unique_invoices', 'unique_streams', 'total_views', 'revenue']

        for d, day in enumerate(dates):
            # current day
            current: np.datetime64 = np.datetime64(day, 'D')

            ## get the target revenue
            plus_30: np.datetime64 = current + np.timedelta64(30, 'D')
            mask = np.in1d(dates, np.arange(current, plus_30, dtype='datetime64[D]'))
            y[d] = df[mask]['revenue'].sum()

            # time-based features
            for variable in variables:
                ## use windows in time back from a specific date
                for num in previous:
                    prev: np.datetime64 = current - np.timedelta64(num, 'D')
                    mask = np.in1d(dates, np.arange(prev, current, dtype='datetime64[D]'))
                    sum_var: float = df[mask][variable].sum()
                    eng_features[f"previous_{num}_{variable}"].append(sum_var)

                ## attempt to capture monthly trend with previous years data (if present)
                start_date: np.datetime64 = current - np.timedelta64(365, 'D')
                stop_date: np.datetime64 = plus_30 - np.timedelta64(365, 'D')
                mask = np.in1d(dates, np.arange(start_date, stop_date, dtype='datetime64[D]'))
                eng_features[f'previous_year_{variable}'].append(df[mask][variable].sum())

            # get type
            mask = np.in1d(dates, np.arange(current, current + 1, dtype='datetime64[D]'))
            var: str = df.loc[mask, "type"].values[0]
            eng_features["type"].append(var)

        X: pd.DataFrame = pd.DataFrame(eng_features)

        # prune data depending on type
        if self.datatype.lower() == 'train':
            dates = dates[X['type'] == 'train']
            y = y[X['type'] == 'train']
            X = X[X['type'] == 'train']
        elif self.datatype.lower() == 'prod':
            dates = dates[X['type'] == 'prod']
            y = y[X['type'] == 'prod']
            X = X[X['type'] == 'prod']
        elif self.datatype.lower() == 'all':
            # all data
            pass
        else:
            # all data
            raise Exception("datatype must be train, prod, or all")

        # drop type
        X = X.drop(columns=['type'])

        # combine features in to df and remove rows with all zeros
        X.fillna(0, inplace=True)
        mask = X.sum(axis=1) > 0
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop=True, inplace=True)

        # remove the last 30 days (because the target is not reliable)
        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop=True, inplace=True)
        # take log of forecated revenue
        if self.log:
            y = np.log(y)

        return X, y, dates
