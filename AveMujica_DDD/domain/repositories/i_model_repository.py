from abc import ABC, abstractmethod
import uuid
from typing import Optional

from ..aggregates.prediction_model import PredictionModel

class IModelRepository(ABC):
    """
    预测模型(PredictionModel)聚合的仓储接口。
    """
    
    @abstractmethod
    def find_by_id(self, model_id: uuid.UUID) -> Optional[PredictionModel]:
        """
        通过ID查找一个预测模型。
        """
        raise NotImplementedError
    
    @abstractmethod
    def save(self, model: PredictionModel) -> None:
        """
        保存一个预测模型。
        """
        raise NotImplementedError 