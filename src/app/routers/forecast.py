import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from ..schemas import ForecastDateOutput, ForecastRangeOutput, MonitorOutput

from fastapi import APIRouter, Depends, HTTPException

from ..pipeline.pipeline import train, load, predict_date, predict_range, monitor_country

router = APIRouter(prefix="/model",
                   responses={404: {"description": "Not found"}}
                   )


# ---------------------------------------------------------------------------------------
# FORECAST MODEL
# ---------------------------------------------------------------------------------------
@router.post("/train/", tags=["v1"])
async def train(name: str):
    name = '.'.join(name.split('.')[:-1] if len(name.split('.')) > 1 else name.split('.')) + '.db'
    model_container = train(name, log=False)
    return {"message": f"model {name} trained"}


@router.get("/train/", tags=["v1"], response_model=Dict[str, float])
async def score(name: str):
    name = '.'.join(name.split('.')[:-1] if len(name.split('.')) > 1 else name.split('.')) + '.db'
    model_container = load(name, log=False)
    scores: Dict[str, float] = model_container.score()
    return scores


@router.get("/forecast_date/", tags=["v1"], response_model=ForecastDateOutput)
async def forecast_date(name: str, country: Optional[str], date: str):
    name = '.'.join(name.split('.')[:-1] if len(name.split('.')) > 1 else name.split('.')) + '.db'
    y_pred = predict_date(name, country, date)
    revenue = float(y_pred[0])
    forecasted_date = pd.to_datetime(date) + pd.DateOffset(days=30)
    initial_date: str = pd.to_datetime(date).strftime('%Y-%m-%d')
    forecasted_date: str = forecasted_date.strftime('%Y-%m-%d')
    return {"country": country, "initial_date": initial_date, "forecasted_date": forecasted_date, "forecasted_revenue": revenue}


@router.get("/forecast_range/", tags=["v1"], response_model=ForecastRangeOutput)
async def forecast_range(name: str, country: Optional[str], initial_date: str, final_date: str):
    name: str = '.'.join(name.split('.')[:-1] if len(name.split('.')) > 1 else name.split('.')) + '.db'
    y_pred: np.ndarray = predict_range(name, country, initial_date, final_date)
    initial_dates = pd.date_range(initial_date, final_date).astype(str).tolist()
    final_dates = pd.date_range(pd.to_datetime(initial_date) + pd.DateOffset(days=30), pd.to_datetime(final_date) + pd.DateOffset(days=30)).astype(str).tolist()
    revenue = [float(y) for y in y_pred]
    return {"country": country, "initial_dates": initial_dates, 'forecasted_dates': final_dates, 'forecasted_revenue': revenue}


@router.get("/monitor/", tags=["v1"], response_model=MonitorOutput)
async def monitor(name: str, country: Optional[str]):
    name: str = '.'.join(name.split('.')[:-1] if len(name.split('.')) > 1 else name.split('.')) + '.db'
    result = monitor_country(name, country, log=False)

    result = {
        'outlier_X': result['outlier_X'],
        'wasserstein_X': result['wasserstein_X'],
        'wasserstein_y': result['wasserstein_y']
    }

    return result
