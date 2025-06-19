from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import List

from .weather_scenario import WeatherScenario
from .prediction_model import PredictionModel

@dataclass
class ForecastDataPoint:
    """
    代表一个预测时间序列中的单个数据点。
    这是一个值对象（Value Object），因为它没有独立的身份。
    """
    timestamp: datetime
    value: float
    upper_bound: float | None = None
    lower_bound: float | None = None

@dataclass
class Forecast:
    """
    预测结果聚合根。
    这是我们领域中最核心的聚合之一，封装了一次完整预测的所有信息。
    """
    # 没有默认值的字段在前
    province: str
    prediction_model: PredictionModel
    matched_weather_scenario: WeatherScenario
    
    # 有默认值的字段在后
    forecast_id: uuid.UUID = field(default_factory=uuid.uuid4)
    creation_time: datetime = field(default_factory=datetime.now)
    time_series: List[ForecastDataPoint] = field(default_factory=list)

    def get_average_forecast(self) -> float:
        """
        一个业务行为：计算本次预测的平均值。
        """
        if not self.time_series:
            return 0.0
        return sum(p.value for p in self.time_series) / len(self.time_series)

    def get_average_interval_width(self) -> float | None:
        """
        一个业务行为：计算平均预测区间的宽度。
        """
        widths = [
            p.upper_bound - p.lower_bound
            for p in self.time_series
            if p.upper_bound is not None and p.lower_bound is not None
        ]
        if not widths:
            return None
        return sum(widths) / len(widths) 