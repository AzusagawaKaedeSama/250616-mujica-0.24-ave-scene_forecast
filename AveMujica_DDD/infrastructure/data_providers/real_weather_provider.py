import os
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Any

from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider


class RealWeatherProvider(IWeatherDataProvider):
    """
    真实的天气数据提供者，从CSV文件中读取实际的天气数据。
    集成现有的数据加载逻辑，支持多省份天气数据。
    """
    
    def __init__(self, data_base_dir: str = 'data'):
        """
        初始化真实天气数据提供者。
        
        Args:
            data_base_dir: 数据根目录，应包含各省份的天气数据文件
        """
        self.data_base_dir = data_base_dir
        self.weather_cache = {}  # 缓存已加载的天气数据
        
        # 定义支持的省份映射（中文名到文件名的映射）
        self.province_mapping = {
            '上海': 'shanghai',
            '安徽': 'anhui', 
            '浙江': 'zhejiang',
            '江苏': 'jiangsu',
            '福建': 'fujian'
        }
        
        print(f"RealWeatherProvider initialized with data directory: {data_base_dir}")

    def get_weather_data_for_date(self, province: str, target_date: date) -> pd.DataFrame:
        """
        获取单日天气数据。
        """
        return self.get_weather_data_for_range(province, target_date, target_date)

    def get_weather_data_for_range(self, province: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        获取指定省份和日期范围的天气数据。
        
        Args:
            province: 省份名称（中文）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含天气数据的DataFrame，索引为datetime，包含温度、湿度等列
        """
        print(f"RealWeatherProvider: Loading weather data for {province} from {start_date} to {end_date}")
        
        # 检查缓存
        cache_key = f"{province}_{start_date}_{end_date}"
        if cache_key in self.weather_cache:
            print(f"Using cached weather data for {cache_key}")
            return self.weather_cache[cache_key]
        
        # 查找对应的天气数据文件
        weather_file = self._find_weather_file(province)
        if not weather_file:
            # 如果找不到真实天气数据，生成模拟数据
            print(f"No real weather data found for {province}, generating synthetic data")
            return self._generate_synthetic_weather_data(start_date, end_date)
        
        try:
            # 加载天气数据文件
            df = pd.read_csv(weather_file, parse_dates=['datetime'])
            df.set_index('datetime', inplace=True)
            
            # 过滤日期范围
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # 确保数据覆盖所需的日期范围
            filtered_df = df.loc[start_datetime:end_datetime]
            
            if filtered_df.empty:
                print(f"No weather data available for the requested date range, generating synthetic data")
                return self._generate_synthetic_weather_data(start_date, end_date)
            
            # 确保包含必需的天气特征列
            required_columns = ['temperature', 'humidity']
            missing_columns = [col for col in required_columns if col not in filtered_df.columns]
            
            if missing_columns:
                print(f"Missing weather columns {missing_columns}, adding synthetic data")
                filtered_df = self._add_missing_weather_features(filtered_df, missing_columns)
            
            # 缓存结果
            self.weather_cache[cache_key] = filtered_df
            
            print(f"Successfully loaded weather data: {len(filtered_df)} records")
            return filtered_df
            
        except Exception as e:
            print(f"Error loading weather data from {weather_file}: {e}")
            print("Falling back to synthetic weather data")
            return self._generate_synthetic_weather_data(start_date, end_date)

    def _find_weather_file(self, province: str) -> str:
        """
        查找省份对应的天气数据文件。
        
        尝试多种可能的文件命名模式：
        1. timeseries_{province}_weather_{year}.csv
        2. timeseries_load_{province}.csv (可能包含天气数据)
        3. weather_{province}.csv
        """
        # 获取英文省份名
        province_en = self.province_mapping.get(province, province.lower())
        
        # 可能的文件名模式
        possible_patterns = [
            f"timeseries_{province}_weather_*.csv",
            f"timeseries_load_{province}.csv", 
            f"timeseries_weather_{province}.csv",
            f"weather_{province}.csv",
            f"{province_en}_weather.csv"
        ]
        
        import glob
        for pattern in possible_patterns:
            search_path = os.path.join(self.data_base_dir, pattern)
            matching_files = glob.glob(search_path)
            if matching_files:
                # 返回第一个匹配的文件
                return matching_files[0]
        
        return None

    def _generate_synthetic_weather_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        生成模拟天气数据，当真实数据不可用时使用。
        """
        # 创建15分钟频率的时间索引
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='15min')
        
        df = pd.DataFrame(index=date_range)
        
        # 生成模拟天气数据（基于季节性模式）
        hour_of_year = (df.index.dayofyear - 1) * 24 + df.index.hour
        
        # 温度：季节性变化 + 日内变化
        df['temperature'] = (
            15 +  # 基础温度
            10 * np.sin(2 * np.pi * hour_of_year / (365 * 24)) +  # 季节性变化
            5 * np.sin(2 * np.pi * df.index.hour / 24)  # 日内变化
        )
        
        # 湿度：反向日内变化
        df['humidity'] = 60 + 20 * np.sin(2 * np.pi * df.index.hour / 24 + np.pi)
        
        # 添加其他常见天气特征
        df['pressure'] = 1013 + 5 * np.sin(2 * np.pi * hour_of_year / (365 * 24))
        df['wind_speed'] = 3 + 2 * np.random.random(len(df))
        df['wind_direction'] = 180 + 90 * np.sin(2 * np.pi * hour_of_year / (365 * 24))
        df['precipitation'] = np.maximum(0, np.random.normal(0, 0.5, len(df)))
        df['solar_radiation'] = np.maximum(0, 
            500 * np.sin(2 * np.pi * df.index.hour / 24) * 
            (1 + 0.3 * np.sin(2 * np.pi * hour_of_year / (365 * 24)))
        )
        
        df.index.name = 'datetime'
        return df

    def _add_missing_weather_features(self, df: pd.DataFrame, missing_columns: list) -> pd.DataFrame:
        """
        为缺失的天气特征列添加合理的默认值或计算值。
        """
        for col in missing_columns:
            if col == 'temperature':
                # 如果没有温度数据，使用季节性模式
                hour_of_year = (df.index.dayofyear - 1) * 24 + df.index.hour
                df[col] = 15 + 10 * np.sin(2 * np.pi * hour_of_year / (365 * 24))
            elif col == 'humidity':
                df[col] = 60 + 20 * np.sin(2 * np.pi * df.index.hour / 24 + np.pi)
            elif col == 'pressure':
                df[col] = 1013.0  # 标准大气压
            elif col == 'wind_speed':
                df[col] = 3.0  # 平均风速
            elif col == 'wind_direction':
                df[col] = 180.0  # 南风
            elif col == 'precipitation':
                df[col] = 0.0  # 无降水
            elif col == 'solar_radiation':
                df[col] = np.maximum(0, 500 * np.sin(2 * np.pi * df.index.hour / 24))
        
        return df

    def clear_cache(self):
        """清除天气数据缓存。"""
        self.weather_cache.clear()
        print("Weather data cache cleared") 