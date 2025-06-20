from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import List

class ForecastType(str, Enum):
    """预测的目标类型"""
    LOAD = "LOAD"   # 负荷预测
    PV = "PV"       # 光伏预测
    WIND = "WIND"   # 风电预测

@dataclass
class PredictionModel:
    """
    预测模型聚合根。
    代表一个训练好的、可用于预测的机器学习模型。
    现在它能自我描述其预测目标和所需的特征。
    """
    name: str
    version: str
    forecast_type: ForecastType
    file_path: str
    target_column: str          # 预测的目标列名, e.g., 'load', 'price'
    feature_columns: List[str]  # 模型依赖的原始特征列名列表
    model_id: uuid.UUID = field(default_factory=uuid.uuid4)
    description: str = ""
    # 可以在这里添加更多元数据，如模型性能指标(mae, rmse等) 