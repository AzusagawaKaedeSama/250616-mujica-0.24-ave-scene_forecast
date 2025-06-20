from datetime import datetime
from uuid import UUID
from typing import List
from pydantic import BaseModel, Field

class ForecastDataPointDTO(BaseModel):
    """
    预测时间序列数据点的数据传输对象。
    继承自Pydantic的BaseModel以获得自动验证和序列化功能。
    """
    timestamp: datetime
    value: float
    upper_bound: float
    lower_bound: float

class ForecastDTO(BaseModel):
    """
    预测结果的数据传输对象。
    """
    forecast_id: UUID
    province: str
    creation_time: datetime
    model_name: str
    scenario_type: str
    time_series: List[ForecastDataPointDTO]

    class Config:
        # Pydantic v2 aribtrary_types_allowed is now a general config setting
        arbitrary_types_allowed = True
        # 禁用保护命名空间以允许model_name字段
        protected_namespaces = () 