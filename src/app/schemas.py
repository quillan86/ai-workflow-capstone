from typing import List, Dict, Optional
from pydantic import BaseModel

class TrainingParameters(BaseModel):
    pass


class PredictionParameters(BaseModel):
    pass


class ForecastDateOutput(BaseModel):
    country: Optional[str]
    initial_date: str
    forecasted_date: str
    forecasted_revenue: float


class ForecastRangeOutput(BaseModel):
    country: Optional[str]
    initial_dates: List[str]
    forecasted_dates: List[str]
    forecasted_revenue: List[float]
