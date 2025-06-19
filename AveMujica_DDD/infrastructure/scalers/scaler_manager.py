import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class ScalerManager:
    """
    一个简洁、健壮的标准化管理器，用于管理模型的训练和预测中的数据缩放。
    """
    def __init__(self, scaler_path: str):
        self.scaler_path = scaler_path
        self.scalers: dict = {}
        os.makedirs(scaler_path, exist_ok=True)

    def transform(self, name: str, data: np.ndarray) -> np.ndarray:
        """使用指定的标准化器转换数据。"""
        scaler = self._get_or_load_scaler(name)
        
        # 记录原始形状，将数据转换为2D进行标准化
        original_shape = data.shape
        data_2d = data.reshape(original_shape[0], -1)
        
        scaled_data_2d = scaler.transform(data_2d)
        
        # 恢复为原始形状
        return scaled_data_2d.reshape(original_shape)

    def inverse_transform(self, name: str, data: np.ndarray) -> np.ndarray:
        """对标准化的数据进行反标准化。"""
        scaler = self._get_or_load_scaler(name)
        
        original_shape = data.shape
        # 预测的输出通常是 (n_samples, 1) 或 (n_samples,)
        data_2d = data.reshape(-1, 1) if len(original_shape) == 1 else data
        
        original_data_2d = scaler.inverse_transform(data_2d)
        
        # 恢复为原始形状
        return original_data_2d.reshape(original_shape)

    def _get_or_load_scaler(self, name: str) -> StandardScaler:
        """获取或从文件加载一个scaler。"""
        if name not in self.scalers:
            path = os.path.join(self.scaler_path, f'{name}_scaler.pkl')
            if not Path(path).exists():
                raise FileNotFoundError(f"标准化器 '{name}' 的文件未找到: {path}")
            
            self.scalers[name] = joblib.load(path)
            print(f"--- [ScalerManager] 从 {path} 加载标准化器 '{name}' ---")
        
        return self.scalers[name] 