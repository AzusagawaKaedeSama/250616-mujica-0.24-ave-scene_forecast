from abc import ABC, abstractmethod
import pandas as pd

from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel

class IPredictionEngine(ABC):
    """
    应用层端口 (Port): 预测引擎接口。

    该接口封装了执行一次模型预测的具体技术实现（例如，加载PyTorch模型并进行推理）。
    """

    @abstractmethod
    def predict(self, model: PredictionModel, input_data: pd.DataFrame) -> pd.Series:
        """
        使用指定的模型和输入数据，执行一次点预测。

        :param model: 要使用的 PredictionModel 聚合实例。
        :param input_data: 经过特征工程处理后，符合模型输入要求的DataFrame。
        :return: 一个包含预测值的 Pandas Series。
        """
        raise NotImplementedError 