"""
训练引擎接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
from ..dtos.training_dto import TrainingProgressDTO
from ...domain.aggregates.training_task import TrainingTask


class ITrainingEngine(ABC):
    """训练引擎接口 - 抽象不同类型的模型训练"""
    
    @abstractmethod
    def prepare_data(self, 
                    task: TrainingTask,
                    data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Returns:
            Tuple[X_train, y_train, X_val, y_val]
        """
        pass
    
    @abstractmethod
    def create_model(self, 
                    task: TrainingTask,
                    input_shape: tuple) -> Any:
        """
        创建模型实例
        
        Args:
            task: 训练任务
            input_shape: 输入数据形状
            
        Returns:
            模型实例
        """
        pass
    
    @abstractmethod
    def train_model(self,
                   task: TrainingTask,
                   model: Any,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   progress_callback: Optional[Callable[[TrainingProgressDTO], None]] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            task: 训练任务
            model: 模型实例
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            progress_callback: 进度回调函数
            
        Returns:
            训练指标字典
        """
        pass
    
    @abstractmethod
    def save_model(self,
                  task: TrainingTask,
                  model: Any) -> str:
        """
        保存模型
        
        Args:
            task: 训练任务
            model: 训练好的模型
            
        Returns:
            模型保存路径
        """
        pass
    
    @abstractmethod
    def evaluate_model(self,
                      task: TrainingTask,
                      model: Any,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            task: 训练任务
            model: 训练好的模型
            X_test, y_test: 测试数据
            
        Returns:
            评估指标字典
        """
        pass
    
    @abstractmethod
    def supports_model_type(self, model_type: str) -> bool:
        """检查是否支持指定的模型类型"""
        pass


class IDataPreprocessor(ABC):
    """数据预处理接口"""
    
    @abstractmethod
    def load_data(self, data_path: str) -> Any:
        """加载原始数据"""
        pass
    
    @abstractmethod
    def engineer_features(self,
                         data: Any,
                         task: TrainingTask) -> Any:
        """特征工程处理"""
        pass
    
    @abstractmethod
    def create_sequences(self,
                        data: Any,
                        task: TrainingTask) -> Tuple[np.ndarray, np.ndarray]:
        """创建时序数据序列"""
        pass
    
    @abstractmethod
    def split_data(self,
                  X: np.ndarray,
                  y: np.ndarray,
                  test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """分割训练和验证数据"""
        pass
    
    @abstractmethod
    def normalize_data(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      task: TrainingTask) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """数据标准化"""
        pass


class IModelPersistence(ABC):
    """模型持久化接口"""
    
    @abstractmethod
    def save_model_files(self,
                        task: TrainingTask,
                        model: Any,
                        additional_files: Optional[Dict[str, Any]] = None) -> str:
        """
        保存模型文件和相关资源
        
        Args:
            task: 训练任务
            model: 模型实例
            additional_files: 额外文件（如标准化器、配置等）
            
        Returns:
            保存目录路径
        """
        pass
    
    @abstractmethod
    def save_training_metadata(self,
                              task: TrainingTask,
                              metrics: Dict[str, Any]) -> None:
        """保存训练元数据"""
        pass
    
    @abstractmethod
    def cleanup_failed_training(self, task: TrainingTask) -> None:
        """清理失败训练的文件"""
        pass 