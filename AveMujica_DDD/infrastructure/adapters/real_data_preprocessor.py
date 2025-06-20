"""
真实数据预处理器实现
"""

import numpy as np
import pandas as pd
from typing import Tuple, Any
from ...application.ports.i_training_engine import IDataPreprocessor
from ...domain.aggregates.training_task import TrainingTask


class RealDataPreprocessor(IDataPreprocessor):
    """真实的数据预处理器实现"""
    
    def load_data(self, data_path: str) -> Any:
        """加载原始数据"""
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    def engineer_features(self, data: Any, task: TrainingTask) -> Any:
        """特征工程处理"""
        # 基础特征工程
        data = data.copy()
        
        # 时间特征
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        
        # 峰谷标记
        if task.is_peak_aware():
            peak_start, peak_end = task.config.peak_hours
            data['is_peak'] = ((data['hour'] >= peak_start) & (data['hour'] <= peak_end)).astype(int)
            
            valley_start, valley_end = task.config.valley_hours
            data['is_valley'] = ((data['hour'] >= valley_start) & (data['hour'] <= valley_end)).astype(int)
        
        return data
    
    def create_sequences(self, data: Any, task: TrainingTask) -> Tuple[np.ndarray, np.ndarray]:
        """创建时序数据序列"""
        seq_length = task.config.seq_length
        
        # 选择特征列
        feature_cols = [col for col in data.columns if col != task.forecast_type.value.lower()]
        target_col = task.forecast_type.value.lower()
        
        # 确保目标列存在
        if target_col not in data.columns:
            target_col = data.columns[0]  # 使用第一列作为目标
        
        values = data[feature_cols].values
        targets = data[target_col].values
        
        X, y = [], []
        for i in range(len(values) - seq_length):
            X.append(values[i:(i + seq_length)])
            y.append(targets[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """分割训练和验证数据"""
        split_idx = int(len(X) * (1 - test_ratio))
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
    
    def normalize_data(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, task: TrainingTask) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """数据标准化"""
        try:
            from utils.scaler_manager import ScalerManager
            
            scaler_manager = ScalerManager(scaler_path=task.get_scaler_directory())
            
            # 标准化X
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            scaler_manager.fit('X', X_train_reshaped)
            X_train_scaled = scaler_manager.transform('X', X_train_reshaped).reshape(X_train.shape)
            
            X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
            X_val_scaled = scaler_manager.transform('X', X_val_reshaped).reshape(X_val.shape)
            
            # 标准化y
            y_train_reshaped = y_train.reshape(-1, 1)
            scaler_manager.fit('y', y_train_reshaped)
            y_train_scaled = scaler_manager.transform('y', y_train_reshaped).flatten()
            
            y_val_reshaped = y_val.reshape(-1, 1)
            y_val_scaled = scaler_manager.transform('y', y_val_reshaped).flatten()
            
            # 保存标准化器
            try:
                scaler_manager.save_scaler('X', scaler_manager.get_scaler('X'))
                scaler_manager.save_scaler('y', scaler_manager.get_scaler('y'))
            except:
                pass  # 忽略保存错误
            
            return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled
            
        except Exception as e:
            print(f"标准化失败，使用原始数据: {e}")
            return X_train, y_train, X_val, y_val 