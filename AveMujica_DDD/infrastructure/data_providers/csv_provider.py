from datetime import date, datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import os

from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider
from AveMujica_DDD.application.ports.i_historical_data_provider import IHistoricalDataProvider

class CsvDataProvider(IWeatherDataProvider, IHistoricalDataProvider):
    """
    一个实现了多个数据提供者接口的类。
    它从单个CSV文件中读取数据，同时为天气预报和历史数据分析提供服务。
    """
    def __init__(self, csv_path: str = "data/timeseries_load_weather_上海.csv"):
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None

    def _load_data(self) -> pd.DataFrame:
        """
        加载数据并返回DataFrame，如果已加载则直接返回。
        此方法保证了返回的一定是一个DataFrame，而不是None。
        """
        if self._df is None:
            print(f"--- [CsvProvider] 首次加载数据从: {self.csv_path} ---")
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"数据文件未找到: {self.csv_path}。请确保文件存在。")
            
            df = pd.read_csv(self.csv_path)
            
            time_col_options = ['datetime', 'DATATIME', 'time']
            time_col = next((col for col in time_col_options if col in df.columns), None)
            
            if time_col is None:
                 raise KeyError(f"在 {self.csv_path} 中找不到任何有效的时间列 (尝试了: {time_col_options})")
            
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
            self._df = df
            print(f"--- [CsvProvider] 数据加载并预处理完成，使用 '{time_col}' 作为时间索引 ---")
        
        return self._df

    def get_weather_data_for_date(self, province: str, target_date: date) -> List[Dict[str, Any]]:
        """获取某一天的天气数据。"""
        df = self._load_data()
        
        start_date = datetime.combine(target_date, datetime.min.time())
        end_date = datetime.combine(target_date, datetime.max.time())
        
        daily_data = df.loc[start_date:end_date]
        
        if daily_data.empty:
            print(f"--- [CsvProvider] 警告: 在 {target_date} 没有找到天气数据，将返回空列表。 ---")
            return []

        weather_cols = ['weather_temperature_c', 'weather_wind_speed', 'weather_relative_humidity', 'weather_precipitation_mm']
        existing_cols = [col for col in weather_cols if col in daily_data.columns]
        
        result_df = daily_data[existing_cols].reset_index()
        result_df = result_df.rename(columns={result_df.columns[0]: 'time'})
        # 为了与PyTorch引擎的输入匹配，重命名列
        result_df = result_df.rename(columns={
            'weather_temperature_c': 'temperature',
            'weather_wind_speed': 'wind_speed',
            'weather_relative_humidity': 'humidity',
            'weather_precipitation_mm': 'precipitation'
        })
        
        return result_df.to_dict('records')

    def get_weather_data_for_range(self, province: str, start_date: date, end_date: date) -> pd.DataFrame:
        """根据日期范围获取天气和负荷数据。"""
        df = self._load_data()
        
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        # 筛选出日期范围内的所有相关数据
        range_data = df.loc[start_datetime:end_datetime]
        
        # 为了与PyTorch引擎的输入匹配，重命名列
        renamed_data = range_data.rename(columns={
            'weather_temperature_c': 'temperature',
            'weather_wind_speed': 'wind_speed',
            'weather_relative_humidity': 'humidity',
            'weather_precipitation_mm': 'precipitation'
        })
        
        return renamed_data

    def get_historical_data(self, province: str) -> pd.DataFrame:
        """获取所有历史数据用于分析。"""
        df = self._load_data()
        
        # 简化列名以供后续使用
        df_renamed = df.rename(columns={'weather_temperature_c': 'temperature'})
        
        required_cols = ['load', 'temperature']
        
        if not all(col in df_renamed.columns for col in required_cols):
            raise KeyError(f"历史数据文件缺少必要的列 (需要 'load' 和 'temperature')。")
        
        # 通过传递一个列表来确保返回的是DataFrame，而不是Series
        result_df = df_renamed[required_cols]
        
        # 显式地告诉类型检查器这是一个DataFrame
        assert isinstance(result_df, pd.DataFrame)
        return result_df 