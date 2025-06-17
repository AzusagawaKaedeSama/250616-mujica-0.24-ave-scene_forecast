from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler

class AbstractTimeSeriesModel(ABC):
    """时间序列模型抽象基类"""
    
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """进行预测"""
        pass
    
    @property
    @abstractmethod
    def input_shape(self):
        """输入形状"""
        pass
    
    @property
    @abstractmethod
    def output_shape(self):
        """输出形状"""
        pass