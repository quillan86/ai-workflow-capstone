# todo - add monitoring
import os
import numpy as np
import pandas as pd
from typing import Union, Dict
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope
from scipy.stats import wasserstein_distance

from .model import ModelContainer


class Monitor:
    def __init__(self, filename: str, log: bool = False, seed: int = 42):
        pipelinepath: str = os.path.dirname(os.path.abspath(__file__))
        self.datadir: str = os.path.abspath(os.path.join(pipelinepath, '..', 'data'))

        self.filename: str = filename
        self.model_container: ModelContainer = ModelContainer(self.datadir, log=log)
        self.model_container.load(filename)
        self.seed: int = seed


    def detect(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Union[float, np.ndarray]]:
        """
        determine outlier and distance thresholds
        return thresholds, outlier model(s) and source distributions for distances
        NOTE: for classification the outlier detection on y is not needed
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values


        contamination: float = 0.01

        xpipe = Pipeline(steps=[('pca', PCA(2, random_state=self.seed)),
                                ('clf', EllipticEnvelope(random_state=self.seed, contamination=contamination))])
        xpipe.fit(X)

        bs_samples: int = 1000
        outliers_X: np.ndarray = np.zeros(bs_samples)
        wasserstein_X: np.ndarray = np.zeros(bs_samples)
        wasserstein_y: np.ndarray = np.zeros(bs_samples)

        for b in range(bs_samples):
            # set random seed
            rng = np.random.default_rng(self.seed + b)

            n_samples = int(np.round(0.80 * X.shape[0]))
            subset_indices = rng.choice(np.arange(X.shape[0]), n_samples, replace=True).astype(int)
            y_bs = y[subset_indices]
            X_bs = X[subset_indices, :]

            test1 = xpipe.predict(X_bs)
            wasserstein_X[b] = wasserstein_distance(X.flatten(), X_bs.flatten())
            wasserstein_y[b] = wasserstein_distance(y, y_bs.flatten())
            outliers_X[b] = 100 * (1.0 - (test1[test1 == 1].size / test1.size))

        ## determine thresholds as a function of the confidence intervals
        outliers_X.sort()
        outlier_X_threshold = outliers_X[int(0.975 * bs_samples)] + outliers_X[int(0.025 * bs_samples)]

        wasserstein_X.sort()
        wasserstein_X_threshold = wasserstein_X[int(0.975 * bs_samples)] + wasserstein_X[int(0.025 * bs_samples)]

        wasserstein_y.sort()
        wasserstein_y_threshold = wasserstein_y[int(0.975 * bs_samples)] + wasserstein_y[int(0.025 * bs_samples)]

        result = {
                    "outlier_X": np.round(outlier_X_threshold, 2),
                    "wasserstein_X": np.round(wasserstein_X_threshold, 2),
                    "wasserstein_y": np.round(wasserstein_y_threshold, 2)
                  }
        return result

    def detect_model(self, country: str) -> Dict[str, Union[float, np.ndarray]]:
        model = self.model_container.models[country]
        X, y, dates = model.load_data(datatype='train')
        result: Dict[str, Union[float, np.ndarray]] = self.detect(X, y)
        return result

    def detect_all(self) -> Dict[str, dict]:
        results: Dict[str, dict] = {}
        for key in self.model_container.models.keys():
            result = self.detect_model(key)
            results[key] = result
        return results

