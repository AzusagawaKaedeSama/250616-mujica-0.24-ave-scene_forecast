"""
训练任务仓储接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..aggregates.training_task import TrainingTask, TrainingStatus, ModelType
from ..aggregates.prediction_model import ForecastType


class ITrainingTaskRepository(ABC):
    """训练任务仓储接口"""
    
    @abstractmethod
    def save(self, task: TrainingTask) -> None:
        """保存训练任务"""
        pass
    
    @abstractmethod
    def find_by_id(self, task_id: str) -> Optional[TrainingTask]:
        """根据ID查找训练任务"""
        pass
    
    @abstractmethod
    def find_by_status(self, status: TrainingStatus) -> List[TrainingTask]:
        """根据状态查找训练任务"""
        pass
    
    @abstractmethod
    def find_by_province_and_type(self, 
                                  province: str, 
                                  model_type: ModelType,
                                  forecast_type: ForecastType) -> List[TrainingTask]:
        """根据省份、模型类型和预测类型查找训练任务"""
        pass
    
    @abstractmethod
    def find_running_tasks(self) -> List[TrainingTask]:
        """查找正在运行的训练任务"""
        pass
    
    @abstractmethod
    def find_recent_tasks(self, limit: int = 10) -> List[TrainingTask]:
        """查找最近的训练任务"""
        pass
    
    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """删除训练任务"""
        pass 