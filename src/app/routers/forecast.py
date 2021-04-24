import pandas as pd
from typing import List, Optional, Dict

from ..schemas import ForecastOutput

from fastapi import APIRouter, Depends, HTTPException

from ..pipeline.pipeline import train, load, predict

router = APIRouter(prefix="/model",
                   responses={404: {"description": "Not found"}}
                   )



# ---------------------------------------------------------------------------------------
# EXTRACT MODEL
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

@router.get("/forecast/", tags=["v1"], response_model=ForecastOutput)
async def forecast(name: str, country: Optional[str], date: str):
    name = '.'.join(name.split('.')[:-1] if len(name.split('.')) > 1 else name.split('.')) + '.db'
    print(name)
    y_pred = predict(name, country, date)
    revenue = float(y_pred[0])
    forecasted_date = pd.to_datetime(date) + pd.DateOffset(days=30)
    initial_date: str = pd.to_datetime(date).strftime('%Y-%m-%d')
    forecasted_date: str = forecasted_date.strftime('%Y-%m-%d')
    return {"country": country, "initial_date": initial_date, "forecasted_date": forecasted_date, "forecasted_revenue": revenue}

