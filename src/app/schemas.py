from typing import List, Optional
from pydantic import BaseModel

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


class MonitorOutput(BaseModel):
    outlier_X: float
    wasserstein_X: float
    wasserstein_y: float