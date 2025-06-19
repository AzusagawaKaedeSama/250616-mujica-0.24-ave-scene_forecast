from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class IModelTrainer(ABC):
    """
    应用层端口 (Port): 模型训练器接口。

    封装了实际执行机器学习模型训练过程的技术细节。
    """

    @abstractmethod
    def train(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """
        执行模型训练。

        :param features: 训练特征。
        :param labels: 训练标签。
        :return: 一个字典，包含训练结果的元数据，例如：
                 {'model_path': '/path/to/model.pth', 'performance_metrics': {'mae': 0.5, 'rmse': 0.8}}
        """
        raise NotImplementedError 