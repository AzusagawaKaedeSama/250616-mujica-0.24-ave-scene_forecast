from abc import ABC, abstractmethod
import pandas as pd

class IFeatureEngineeringService(ABC):
    """
    应用层端口 (Port): 特征工程服务接口。

    封装了为模型训练和预测准备特征的复杂逻辑。
    """

    @abstractmethod
    def preprocess_for_training(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        为模型训练准备特征和标签。

        :param data: 原始的历史数据。
        :return: 一个元组，包含特征 (X) 和标签 (y)。
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess_for_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        为预测准备特征。

        :param data: 原始的输入数据 (例如，一天的天气数据)。
        :return: 经过特征工程处理后，符合模型输入要求的DataFrame。
        """
        raise NotImplementedError 