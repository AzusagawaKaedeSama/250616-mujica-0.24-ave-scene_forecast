from dataclasses import dataclass
import uuid
from datetime import datetime
from typing import List, Optional

@dataclass(frozen=True)
class ForecastDataPointDTO:
    """
    单个预测数据点的DTO。
    `frozen=True` 意味着它是不可变的，保证了数据的纯粹性。
    """
    timestamp: datetime
    value: float
    upper_bound: Optional[float]
    lower_bound: Optional[float]

@dataclass(frozen=True)
class ForecastDTO:
    """
    用于向接口层返回预测结果的DTO。
    它只包含数据，没有任何业务逻辑。
    """
    forecast_id: uuid.UUID
    province: str
    creation_time: datetime
    model_name: str
    scenario_type: str
    time_series: List[ForecastDataPointDTO] 