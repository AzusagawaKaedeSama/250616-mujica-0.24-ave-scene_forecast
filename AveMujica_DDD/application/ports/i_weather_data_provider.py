from abc import ABC, abstractmethod
from datetime import date
from typing import List, Dict, Any
import pandas as pd

class IWeatherDataProvider(ABC):
    """
    应用层端口 (Port): 天气数据提供者接口。

    该接口定义了应用层如何从外部世界（基础设施层）获取天气数据。
    应用服务将依赖此接口，而不是某个具体的文件读取或API调用实现。
    """

    @abstractmethod
    def get_weather_data_for_date(self, province: str, target_date: date) -> List[Dict[str, Any]]:
        """
        根据省份和目标日期，获取一整天的天气预报数据。
        
        :param province: 目标省份，例如 "上海"。
        :param target_date: 目标预测日期。
        :return: 一个字典列表，每个字典代表一个时间点（例如一小时）的天气数据。
                 例如: [{'time': '2024-07-10 00:00:00', 'temperature': 28.5, 'humidity': 85}, ...]
        """
        raise NotImplementedError

    @abstractmethod
    def get_weather_data_for_range(self, province: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        根据省份和日期范围，获取天气数据。
        
        :param province: 目标省份。
        :param start_date: 开始日期。
        :param end_date: 结束日期。
        :return: 一个包含天气数据的DataFrame，以datetime为索引。
        """
        raise NotImplementedError 