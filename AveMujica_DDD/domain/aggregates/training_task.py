"""
训练任务聚合根 - 管理模型训练的完整生命周期
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

from ..aggregates.prediction_model import ForecastType


class TrainingStatus(Enum):
    """训练状态枚举"""
    PENDING = "pending"          # 待开始
    RUNNING = "running"          # 进行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"      # 已取消


class ModelType(Enum):
    """模型类型枚举"""
    CONVTRANS = "convtrans"                    # 通用ConvTransformer
    CONVTRANS_PEAK = "convtrans_peak"          # 峰谷感知
    CONVTRANS_WEATHER = "convtrans_weather"    # 天气感知
    PROBABILISTIC = "probabilistic"            # 概率预测
    INTERVAL = "interval"                      # 区间预测


@dataclass
class TrainingConfig:
    """训练配置值对象"""
    seq_length: int = 96
    pred_length: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 50
    patience: int = 10
    
    # 峰谷感知配置
    peak_hours: tuple = (8, 20)
    valley_hours: tuple = (0, 6)
    peak_weight: float = 2.5
    valley_weight: float = 1.5
    
    # 天气感知配置
    weather_features: Optional[List[str]] = None
    
    # 概率预测配置
    quantiles: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'seq_length': self.seq_length,
            'pred_length': self.pred_length,
            'batch_size': self.batch_size,
            'lr': self.learning_rate,
            'epochs': self.epochs,
            'patience': self.patience,
            'peak_hours': self.peak_hours,
            'valley_hours': self.valley_hours,
            'peak_weight': self.peak_weight,
            'valley_weight': self.valley_weight,
            'use_peak_loss': True,
            'weather_features': self.weather_features,
            'quantiles': self.quantiles
        }


@dataclass
class TrainingMetrics:
    """训练指标值对象"""
    final_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    epochs_trained: Optional[int] = None
    training_time_seconds: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    
    def update_metrics(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class TrainingTask:
    """训练任务聚合根"""
    
    def __init__(self, 
                 task_id: Optional[str] = None,
                 model_type: ModelType = ModelType.CONVTRANS,
                 forecast_type: ForecastType = ForecastType.LOAD,
                 province: str = "上海",
                 train_start_date: Optional[date] = None,
                 train_end_date: Optional[date] = None,
                 config: Optional[TrainingConfig] = None,
                 retrain: bool = False):
        
        self.task_id = task_id or str(uuid.uuid4())
        self.model_type = model_type
        self.forecast_type = forecast_type
        self.province = province
        self.train_start_date = train_start_date or date(2024, 1, 1)
        self.train_end_date = train_end_date or date(2024, 3, 31)
        self.config = config or TrainingConfig()
        self.retrain = retrain
        
        # 状态管理
        self.status = TrainingStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        
        # 训练结果
        self.metrics = TrainingMetrics()
        self.model_path: Optional[str] = None
        self.logs: List[str] = []
    
    def start_training(self) -> None:
        """开始训练"""
        if self.status != TrainingStatus.PENDING:
            raise ValueError(f"任务状态必须为PENDING才能开始训练，当前状态：{self.status}")
        
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.now()
        self.add_log("训练任务开始")
    
    def complete_training(self, 
                         model_path: str, 
                         final_metrics: Dict[str, Any]) -> None:
        """完成训练"""
        if self.status != TrainingStatus.RUNNING:
            raise ValueError(f"任务状态必须为RUNNING才能完成训练，当前状态：{self.status}")
        
        self.status = TrainingStatus.COMPLETED
        self.completed_at = datetime.now()
        self.model_path = model_path
        self.metrics.update_metrics(**final_metrics)
        self.add_log("训练任务完成")
    
    def fail_training(self, error_message: str) -> None:
        """训练失败"""
        if self.status not in [TrainingStatus.RUNNING, TrainingStatus.PENDING]:
            raise ValueError(f"无法将任务状态设置为FAILED，当前状态：{self.status}")
        
        self.status = TrainingStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        self.add_log(f"训练任务失败：{error_message}")
    
    def cancel_training(self) -> None:
        """取消训练"""
        if self.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
            raise ValueError(f"无法取消已完成或失败的任务，当前状态：{self.status}")
        
        self.status = TrainingStatus.CANCELLED
        self.completed_at = datetime.now()
        self.add_log("训练任务已取消")
    
    def add_log(self, message: str) -> None:
        """添加日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
    
    def update_progress(self, epoch: int, loss: float, val_loss: Optional[float] = None) -> None:
        """更新训练进度"""
        self.metrics.epochs_trained = epoch
        self.metrics.final_loss = loss
        if val_loss is not None:
            self.metrics.best_val_loss = val_loss
        
        self.add_log(f"Epoch {epoch}: loss={loss:.4f}" + 
                    (f", val_loss={val_loss:.4f}" if val_loss else ""))
    
    def is_weather_aware(self) -> bool:
        """是否为天气感知训练"""
        return self.model_type == ModelType.CONVTRANS_WEATHER
    
    def is_peak_aware(self) -> bool:
        """是否为峰谷感知训练"""
        return self.model_type in [ModelType.CONVTRANS_PEAK, ModelType.INTERVAL]
    
    def is_probabilistic(self) -> bool:
        """是否为概率预测训练"""
        return self.model_type == ModelType.PROBABILISTIC
    
    def get_model_directory(self) -> str:
        """获取模型保存目录"""
        return f"models/{self.model_type.value}/{self.forecast_type.value}/{self.province}"
    
    def get_scaler_directory(self) -> str:
        """获取标准化器保存目录"""
        return f"models/scalers/{self.model_type.value}/{self.forecast_type.value}/{self.province}"
    
    def should_retrain(self) -> bool:
        """是否应该重新训练"""
        return self.retrain
    
    def get_training_duration(self) -> Optional[float]:
        """获取训练时长（秒）"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'task_id': self.task_id,
            'model_type': self.model_type.value,
            'forecast_type': self.forecast_type.value,
            'province': self.province,
            'train_start_date': self.train_start_date.isoformat(),
            'train_end_date': self.train_end_date.isoformat(),
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'model_path': self.model_path,
            'config': self.config.to_dict(),
            'metrics': {
                'final_loss': self.metrics.final_loss,
                'best_val_loss': self.metrics.best_val_loss,
                'epochs_trained': self.metrics.epochs_trained,
                'training_time_seconds': self.get_training_duration(),
                'mae': self.metrics.mae,
                'rmse': self.metrics.rmse,
                'mape': self.metrics.mape
            },
            'logs': self.logs[-10:]  # 只返回最近10条日志
        } 