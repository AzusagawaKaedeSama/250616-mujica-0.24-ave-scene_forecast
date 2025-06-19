from dataclasses import dataclass
import uuid
from typing import Dict, Any

@dataclass(frozen=True)
class ModelTrainingDTO:
    """
    用于向接口层返回模型训练结果的DTO。
    """
    model_id: uuid.UUID
    model_name: str
    model_version: str
    message: str
    performance_metrics: Dict[str, Any] 