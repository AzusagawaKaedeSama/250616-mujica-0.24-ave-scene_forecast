"""
训练相关的数据传输对象
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from ...domain.aggregates.training_task import ModelType, TrainingStatus
from ...domain.aggregates.prediction_model import ForecastType


@dataclass
class TrainingRequestDTO:
    """训练请求DTO"""
    model_type: str = "convtrans"
    forecast_type: str = "load"
    province: str = "上海"
    train_start_date: str = "2024-01-01"
    train_end_date: str = "2024-03-31"
    
    # 训练配置
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 10
    retrain: bool = False
    
    # 峰谷感知配置
    peak_hours: tuple = (8, 20)
    valley_hours: tuple = (0, 6)
    peak_weight: float = 2.5
    valley_weight: float = 1.5
    
    # 天气感知配置
    weather_features: Optional[List[str]] = None
    weather_data_path: Optional[str] = None
    
    # 概率预测配置
    quantiles: Optional[List[float]] = None
    
    def to_model_type(self) -> ModelType:
        """转换为模型类型枚举"""
        type_mapping = {
            "convtrans": ModelType.CONVTRANS,
            "convtrans_peak": ModelType.CONVTRANS_PEAK,
            "convtrans_weather": ModelType.CONVTRANS_WEATHER,
            "probabilistic": ModelType.PROBABILISTIC,
            "interval": ModelType.INTERVAL
        }
        return type_mapping.get(self.model_type, ModelType.CONVTRANS)
    
    def to_forecast_type(self) -> ForecastType:
        """转换为预测类型枚举"""
        type_mapping = {
            "load": ForecastType.LOAD,
            "pv": ForecastType.PV,
            "wind": ForecastType.WIND
        }
        return type_mapping.get(self.forecast_type, ForecastType.LOAD)


@dataclass
class TrainingProgressDTO:
    """训练进度DTO"""
    task_id: str
    epoch: int
    total_epochs: int
    current_loss: float
    best_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    estimated_remaining_time: Optional[float] = None  # 秒
    
    def get_progress_percentage(self) -> float:
        """获取进度百分比"""
        return (self.epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0.0


@dataclass
class TrainingResultDTO:
    """训练结果DTO"""
    task_id: str
    status: str
    model_type: str
    forecast_type: str
    province: str
    
    # 时间信息
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    
    # 训练配置
    config: Optional[Dict[str, Any]] = None
    
    # 训练指标
    final_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    epochs_trained: Optional[int] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    
    # 模型信息
    model_path: Optional[str] = None
    model_size_mb: Optional[float] = None
    
    # 错误信息
    error_message: Optional[str] = None
    logs: Optional[List[str]] = None
    
    @classmethod
    def from_training_task(cls, task) -> 'TrainingResultDTO':
        """从训练任务创建DTO"""
        task_dict = task.to_dict()
        
        return cls(
            task_id=task_dict['task_id'],
            status=task_dict['status'],
            model_type=task_dict['model_type'],
            forecast_type=task_dict['forecast_type'],
            province=task_dict['province'],
            created_at=task_dict['created_at'],
            started_at=task_dict.get('started_at'),
            completed_at=task_dict.get('completed_at'),
            training_duration_seconds=task_dict['metrics'].get('training_time_seconds'),
            config=task_dict['config'],
            final_loss=task_dict['metrics'].get('final_loss'),
            best_val_loss=task_dict['metrics'].get('best_val_loss'),
            epochs_trained=task_dict['metrics'].get('epochs_trained'),
            mae=task_dict['metrics'].get('mae'),
            rmse=task_dict['metrics'].get('rmse'),
            mape=task_dict['metrics'].get('mape'),
            model_path=task_dict['model_path'],
            error_message=task_dict['error_message'],
            logs=task_dict['logs']
        )


@dataclass
class TrainingHistoryDTO:
    """训练历史DTO"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    running_tasks: int
    recent_tasks: List[TrainingResultDTO]
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        total = self.completed_tasks + self.failed_tasks
        return (self.completed_tasks / total * 100) if total > 0 else 0.0 