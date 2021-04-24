import os
import shelve
import pickle
from typing import Optional, Union, List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer

from .state import State
from .features import FeatureEngineer


class Model:
    def __init__(self, datadir: str, country: Optional[str] = None, log: bool = False, seed: int = 42):
        """
        A single model per country.
        """
        self.datadir: str = datadir
        self.country: str = country
        self.log: bool = log
        self.state: State = State(datadir)
        self.seed: int = seed

        # model initializations
        self.reg: Optional[RandomizedSearchCV] = None # model inside random CV
        self.estimator: Optional[RandomForestRegressor] = None # the optimized model itself

    def load_data(self, datatype: str ='all'):
        dfi = self.state.load(country=self.country, datatype ='all', droptype=False)
        feature_engineer: FeatureEngineer = FeatureEngineer(datatype=datatype, log=self.log)
        X, y, dates = feature_engineer.run(dfi)
        return X, y, dates

    def load_train_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load training data.
        """
        X_train, y_train, dates_train = self.load_data('train')
        return X_train, y_train, dates_train

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: np.ndarray) -> None:
        """
        Fit a model.
        X_train: Features to train over
        y_train: target to train over (revenue or log revenue)
        """
        # Hyperparameter optimization

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf: RandomForestRegressor = RandomForestRegressor(random_state=self.seed)

        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        self.reg = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,
                                      cv=3, verbose=1, random_state=self.seed, n_jobs=-1)

        # Fit the random search model
        self.reg.fit(X_train, y_train)
        self.estimator = self.reg.best_estimator_
        return

    def date_to_features(self, date: str):
        X_prod, y_prod, dates_prod = self.load_data('all')
        date = pd.to_datetime(date)
        X_data = X_prod.loc[dates_prod == date, :]
        return X_data

    def date_range_to_features(self, initial_date: str, final_date: str):
        X_prod, y_prod, dates_prod = self.load_data('all')
        initial_date = pd.to_datetime(initial_date)
        final_date = pd.to_datetime(final_date)
        X_data = X_prod.loc[(dates_prod >= initial_date) & (dates_prod <= final_date), :]
        return X_data

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict revenue. If data is trained on log of data, convert back to revenue.
        """
        y_pred: np.ndarray = self.reg.predict(X)
        if self.log:
            r_pred: np.ndarray = np.exp(y_pred)
        else:
            r_pred: np.ndarray = y_pred
        return r_pred

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> Tuple[float, float]:
        """
        Cross validated RMSE
        """
        cv = KFold(5, random_state=self.seed, shuffle=True) # for reduplication
        cv_score = cross_val_score(self.estimator, X, y, cv=cv, scoring=make_scorer(mean_squared_error))
        cv_score = np.array([np.sqrt(s) for s in cv_score])
        cv_score_mean = cv_score.mean()
        cv_score_std = cv_score.std()
        return cv_score_mean, cv_score_std

    def save_object(self) -> Dict[str, Union[str, int, RandomForestRegressor, RandomizedSearchCV, None]]:
        # put in an object to be read by dictionary
        save_container: Dict[str, Union[str, int, RandomForestRegressor, RandomizedSearchCV, None]] = {}
        save_container['datadir'] = self.datadir
        save_container['country'] = self.country
        save_container['log'] = self.log
        save_container['reg'] = self.reg
        save_container['estimator'] = self.estimator
        save_container['seed'] = self.seed
        return save_container

    def save(self, filename: str) -> None:
        # pickle this particular model rather than shelving it
        save_container = self.save_object()
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump(save_container, f)
        return

    def load_estimator(self, save_container: Dict[str, Union[str, int, RandomForestRegressor, RandomizedSearchCV]]) -> None:
        self.reg: Optional[RandomizedSearchCV] = save_container['reg']
        self.estimator: Optional[RandomForestRegressor] = save_container['estimator']
        return


class ModelContainer:

    def __init__(self, datadir: str, log: bool = False, seed: int = 42):
        """
        Container for models - one model per country.
        """
        self.datadir: str = datadir
        self.train_datadir = os.path.join(self.datadir, "cs-train")
        self.prod_datadir = os.path.join(self.datadir, "cs-production")
        self.ts_datadir = os.path.join(self.datadir, "ts-data")
        self.model_datadir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
        self.log: bool = log
        self.seed: int = seed
        self.models: Dict[str, Model] = {}

    def _get_top_countries(self, N: int = 15) -> List[Optional[str]]:
        def get_file(file, country):
            df = pd.read_csv(file)
            df['country'] = country
            return df
        # only on training data
        files = [os.path.join(self.ts_datadir, f) for f in os.listdir(self.ts_datadir) if os.path.isfile(os.path.join(self.ts_datadir, f))]
        countries = [f.split('.')[0] for f in os.listdir(self.ts_datadir) if
                 os.path.isfile(os.path.join(self.ts_datadir, f))]

        df = pd.concat(get_file(file, country) for file, country in zip(files, countries))
        df = df[df['type'] == 'train']
        # find the top N countries (wrt revenue)
        table: pd.DataFrame = pd.pivot_table(df, index='country', values="revenue", aggfunc='sum')
        table.columns = ['total_revenue']
        table.sort_values(by='total_revenue', inplace=True, ascending=False)
        top_countries: List[Optional[str]] = np.array(list(table.index))[:N].tolist()
        return top_countries

    def train(self, filename, N: int = 10) -> Dict[str, Model]:
        """
        Train over the top 15 countries
        """
        # absolute path of filename
        filename = os.path.join(self.model_datadir, filename)
        # get top countries to iterate over
        top_countries: List[Optional[str]] = self._get_top_countries(N=N)

        # delete shelf if it already exists
        if os.path.exists(filename):
            os.remove(filename)

        self.models = {}
        with shelve.open(filename) as db:
            # iterate overall countries plus all ("None" here)
            for country in [None] + top_countries:
                # create model
                model = Model(self.datadir, country, log=self.log, seed=self.seed)
                # train model
                print(f"*** TRAINING {country} ***")
                X_train, y_train, dates_train = model.load_train_data()
                model.fit(X_train, y_train)
                # save model - special one for none
                if country is None:
                    key: Optional[str] = 'all'
                else:
                    key: Optional[str] = country
                save_container: Dict[str, Union[str, int, RandomForestRegressor, RandomizedSearchCV, None]] = model.save_object()
                db[key] = save_container
                self.models[key] = model
                del X_train, y_train, dates_train
                print(f"*** SAVED {key} ***")
        return self.models

    def load(self, filename: str, N: int = 10) -> Dict[str, Model]:
        """
        Load models into memory.
        """
        # absolute path of filename
        print(self.model_datadir)
        filename = os.path.join(self.model_datadir, filename)
        with shelve.open(filename) as db:
            keylist: List[str] = self._get_top_countries(N=N)
            self.models: Dict[str, Model] = {}
            for key in keylist:
                save_container = db[key]
                model = Model(save_container['datadir'], save_container['country'], log=save_container['log'], seed=save_container['seed'])
                model.load_estimator(save_container)
                self.models[key] = model
        return self.models

    def score(self) -> Dict[str, float]:
        """
        Obtain training scores for model.
        """
        scores = {}
        for country in self.models:
            model: Model = self.models[country]
            # load model data
            X_train, y_train, dates_train = model.load_train_data()
            del dates_train # not needed
            rmse, rmse_std = model.score(X_train, y_train)
            print(f"{country} RMSE: {rmse:.2f} +/- {rmse_std:.2f}")
            scores[country] = rmse
        return scores

    def predict_date(self, country: Optional[str], date: str) -> np.ndarray:
        model: Model = self.models[country]
        X = model.date_to_features(date)
        r_pred = model.predict(X)
        return r_pred

    def predict_range(self, country: Optional[str], initial_date: str, final_date: str) -> np.ndarray:
        model: Model = self.models[country]
        X = model.date_range_to_features(initial_date, final_date)
        r_pred = model.predict(X)
        return r_pred
