from abc import ABC, abstractmethod
import pandas as pd

class IHistoricalDataProvider(ABC):
    """
    应用层端口 (Port): 历史数据提供者接口。

    定义了应用层如何获取用于分析和训练的历史时间序列数据。
    """

    @abstractmethod
    def get_historical_data(self, province: str) -> pd.DataFrame:
        """
        根据省份获取完整的历史天气和负荷数据。

        :param province: 目标省份。
        :return: 一个包含多列（如时间、温度、负荷等）的Pandas DataFrame。
        """
        raise NotImplementedError 