from typing import List, Dict, Optional
from pydantic import BaseModel

class TrainingParameters(BaseModel):
    pass


class PredictionParameters(BaseModel):
    pass


class ForecastOutput(BaseModel):
    country: Optional[str]
    initial_date: str
    forecasted_date: str
    forecasted_revenue: float
