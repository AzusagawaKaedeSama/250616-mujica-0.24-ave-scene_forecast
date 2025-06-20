from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider


class DummyWeatherProvider(IWeatherDataProvider):
    """
    一个假的IWeatherDataProvider实现，用于提供测试所需的天气数据。
    它会生成一个符合日期范围和15分钟间隔的、带有模拟值的DataFrame。
    """
    def get_weather_data_for_date(self, province: str, target_date: date) -> pd.DataFrame:
        """
        获取单日天气数据，通过调用范围方法实现。
        """
        return self.get_weather_data_for_range(province, target_date, target_date)

    def get_weather_data_for_range(self, province: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        生成从开始日期到结束日期的模拟天气数据。
        """
        print(f"DummyWeatherProvider: Generating data for {province} from {start_date} to {end_date}")
        
        # 创建一个15分钟频率的日期时间索引
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='15min')
        
        # 创建一个空的DataFrame
        df = pd.DataFrame(index=date_range)
        
        # 生成模拟数据
        # 使用三角函数来模拟日内和季节性温度变化
        hour_of_year = (df.index.dayofyear - 1) * 24 + df.index.hour
        df['temperature'] = (
            15 +  # Base temperature
            10 * np.sin(2 * np.pi * hour_of_year / (365 * 24)) +  # Seasonal variation
            5 * np.sin(2 * np.pi * df.index.hour / 24)  # Daily variation
        )
        df['humidity'] = 60 + 20 * np.sin(2 * np.pi * df.index.hour / 24 + np.pi)
        
        df.index.name = 'datetime'
        return df 