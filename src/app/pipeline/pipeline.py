import pandas as pd
import numpy as np
import os
from .model import ModelContainer
from typing import Optional, Union

pipelinepath: str = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.abspath(os.path.join(pipelinepath, '..', 'data'))


def train(model_name: str, N: int = 10, log: bool = False) -> ModelContainer:
    model_container = ModelContainer(datadir, log=log)
    print(model_container.datadir)
    model_container.train(model_name, N=N)
    return model_container


def load(filename: str, log: bool = False) -> ModelContainer:
    model_container = ModelContainer(datadir, log=log)
    model_container.load(filename)
    return model_container

def predict_date(filename: str, country: Optional[str], date: str) -> np.ndarray:
    model_container = load(filename)
    y_pred = model_container.predict_date(country, date)
    return y_pred

def predict_range(filename: str, country: Optional[str], initial_date: str, final_date: str) -> np.ndarray:
    model_container = load(filename)
    y_pred = model_container.predict_range(country, initial_date, final_date)
    return y_pred


if __name__ == "__main__":
    print(datadir)
    load("initial_model")
