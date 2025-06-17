import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WeatherProcessor:
    """
    处理和管理气象数据以用于负荷预测
    """
    
    def __init__(self, weather_data_dir=None):
        """
        初始化气象数据处理器
        
        参数:
        weather_data_dir (str): 气象数据目录
        """
        self.weather_data_dir = weather_data_dir if weather_data_dir else 'data_preprocess'
        self.weather_data = None
        self.load_data = None
    
    def load_weather_data(self, location=None, year_month=None):
        """
        加载特定位置和时间的气象数据
        
        参数:
        location (str): 位置标识符 (如 'fuzhou_lon119.30_lat26.08')
        year_month (str): 年月，格式为 'YYYY-MM'
        
        返回:
        DataFrame: 处理后的气象数据
        """
        if location is None:
            # 默认使用福州数据
            location = 'fuzhou_lon119.30_lat26.08'
        
        if year_month is None:
            # 使用当前年月
            year_month = datetime.now().strftime('%Y-%m')
        
        # 构建文件路径模式
        file_pattern = f"{self.weather_data_dir}/{location}/{year_month}/*.csv"
        
        # 查找匹配的文件
        import glob
        matching_files = glob.glob(file_pattern)
        
        if not matching_files:
            # 尝试直接加载特定文件
            direct_file = f"{self.weather_data_dir}/{location}/{year_month}/reanalysis-era5-single-levels-timeseries-sfcnkachzo5.csv"
            if os.path.exists(direct_file):
                matching_files = [direct_file]
        
        if not matching_files:
            raise FileNotFoundError(f"未找到匹配的气象数据文件: {file_pattern}")
        
        # 加载并合并所有匹配的文件
        dfs = []
        for file in matching_files:
            try:
                df = pd.read_csv(file)
                # 确保时间列是日期时间格式
                if 'valid_time' in df.columns:
                    df['valid_time'] = pd.to_datetime(df['valid_time'])
                    df = df.set_index('valid_time')
                dfs.append(df)
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")
        
        if not dfs:
            raise ValueError("无法加载任何气象数据文件")
        
        # 合并所有数据框
        if len(dfs) > 1:
            weather_data = pd.concat(dfs, axis=0)
            # 按时间排序并删除重复项
            weather_data = weather_data.sort_index().drop_duplicates()
        else:
            weather_data = dfs[0]
        
        self.weather_data = weather_data
        return weather_data
    
    def load_multiple_months(self, location=None, start_year_month=None, end_year_month=None):
        """
        加载多个月份的气象数据
        
        参数:
        location (str): 位置标识符
        start_year_month (str): 开始年月，格式为 'YYYY-MM'
        end_year_month (str): 结束年月，格式为 'YYYY-MM'
        
        返回:
        DataFrame: 合并后的气象数据
        """
        if start_year_month is None or end_year_month is None:
            raise ValueError("必须指定开始和结束年月")
        
        # 解析开始和结束日期
        start_date = datetime.strptime(start_year_month, '%Y-%m')
        end_date = datetime.strptime(end_year_month, '%Y-%m')
        
        # 创建月份列表
        current_date = start_date
        month_list = []
        
        while current_date <= end_date:
            month_list.append(current_date.strftime('%Y-%m'))
            # 移动到下一个月
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        # 加载每个月的数据
        all_weather_data = []
        for year_month in month_list:
            try:
                weather_data = self.load_weather_data(location, year_month)
                all_weather_data.append(weather_data)
            except FileNotFoundError:
                print(f"警告: 未找到 {year_month} 的气象数据")
        
        if not all_weather_data:
            raise ValueError("未能加载任何气象数据")
        
        # 合并所有数据
        merged_data = pd.concat(all_weather_data, axis=0)
        merged_data = merged_data.sort_index().drop_duplicates()
        
        self.weather_data = merged_data
        return merged_data
    
    def preprocess_weather_data(self):
        """
        预处理气象数据，计算有用的衍生特征
        
        返回:
        DataFrame: 预处理后的气象数据
        """
        if self.weather_data is None:
            raise ValueError("请先加载气象数据")
        
        # 创建副本以避免修改原始数据
        processed_data = self.weather_data.copy()
        
        # 计算风速 (由u10和v10风向分量计算)
        if 'u10' in processed_data.columns and 'v10' in processed_data.columns:
            processed_data['wind_speed'] = np.sqrt(processed_data['u10']**2 + processed_data['v10']**2)
        
        # 将开尔文温度转换为摄氏度
        if 't2m' in processed_data.columns:
            processed_data['temperature_c'] = processed_data['t2m'] - 273.15
        
        if 'd2m' in processed_data.columns:
            processed_data['dew_point_c'] = processed_data['d2m'] - 273.15
        
        # 计算相对湿度 (基于温度和露点)
        if 'temperature_c' in processed_data.columns and 'dew_point_c' in processed_data.columns:
            # 近似计算相对湿度
            processed_data['relative_humidity'] = 100 * (
                np.exp((17.625 * processed_data['dew_point_c']) / (243.04 + processed_data['dew_point_c'])) /
                np.exp((17.625 * processed_data['temperature_c']) / (243.04 + processed_data['temperature_c']))
            )
        
        # 添加时间特征
        processed_data['hour'] = processed_data.index.hour
        processed_data['day_of_week'] = processed_data.index.dayofweek
        processed_data['month'] = processed_data.index.month
        processed_data['is_weekend'] = (processed_data.index.dayofweek >= 5).astype(int)
        
        # 温度变化率
        if 'temperature_c' in processed_data.columns:
            processed_data['temp_change_rate'] = processed_data['temperature_c'].diff()
        
        # 计算HDD (供暖度日) 和 CDD (制冷度日)
        if 'temperature_c' in processed_data.columns:
            # 基准温度通常为18℃
            base_temp = 18.0
            processed_data['HDD'] = np.maximum(0, base_temp - processed_data['temperature_c'])
            processed_data['CDD'] = np.maximum(0, processed_data['temperature_c'] - base_temp)
        
        # 填充缺失值
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        
        return processed_data
    
    def merge_with_load_data(self, load_data, interpolate_method='linear'):
        """
        将气象数据与负荷数据合并
        
        参数:
        load_data (DataFrame): 包含时间索引的负荷数据
        interpolate_method (str): 用于插值的方法，默认为'linear'
        
        返回:
        DataFrame: 合并后的数据框
        """
        if self.weather_data is None:
            raise ValueError("请先加载气象数据")
        
        # 预处理气象数据
        weather_data = self.preprocess_weather_data()
        
        # 确保负荷数据有日期时间索引
        if not isinstance(load_data.index, pd.DatetimeIndex):
            raise ValueError("负荷数据必须有日期时间索引")
        
        # 重新采样气象数据以匹配负荷数据的频率
        load_freq = pd.infer_freq(load_data.index)
        if load_freq:
            # 使用重采样和插值
            weather_resampled = weather_data.resample(load_freq).interpolate(method=interpolate_method)
        else:
            # 如果无法推断频率，使用时间点合并
            common_times = load_data.index.intersection(weather_data.index)
            weather_resampled = weather_data.loc[common_times]
            print(f"警告: 无法推断负荷数据的频率，使用时间点交集 (共 {len(common_times)} 点)")
        
        # 确保所有负荷时间点都有对应的气象数据
        missing_times = load_data.index.difference(weather_resampled.index)
        if len(missing_times) > 0:
            print(f"警告: {len(missing_times)} 个负荷数据时间点没有对应的气象数据")
            
            # 为缺失的时间点创建插值气象数据
            missing_df = pd.DataFrame(index=missing_times)
            weather_full = pd.concat([weather_resampled, missing_df]).sort_index()
            weather_full = weather_full.interpolate(method=interpolate_method)
            weather_resampled = weather_full
        
        # 合并数据
        merged_data = load_data.copy()
        
        # 选择要添加的气象特征列
        weather_features = [
            'temperature_c', 'wind_speed', 'relative_humidity', 
            'HDD', 'CDD', 'temp_change_rate'
        ]
        
        # 确保选择的列在气象数据中存在
        existing_features = [col for col in weather_features if col in weather_resampled.columns]
        
        # 添加气象特征到负荷数据
        for feature in existing_features:
            merged_data[f'weather_{feature}'] = weather_resampled[feature]
        
        self.load_data = merged_data
        return merged_data
    
    def save_processed_data(self, output_path=None, dataset_id='fujian'):
        """
        保存处理后的数据
        
        参数:
        output_path (str): 输出路径
        dataset_id (str): 数据集标识符
        
        返回:
        str: 保存的文件路径
        """
        if self.load_data is None:
            raise ValueError("没有要保存的处理后数据")
        
        if output_path is None:
            # 默认保存路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"data/timeseries_load_weather_{dataset_id}_{timestamp}.csv"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存数据
        self.load_data.to_csv(output_path)
        print(f"处理后的数据已保存到 {output_path}")
        
        return output_path 