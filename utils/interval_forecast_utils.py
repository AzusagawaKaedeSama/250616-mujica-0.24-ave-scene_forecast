"""
区间预测工具集 - 合并多个工具类和函数
支持区间预测模型的训练和预测功能
"""

import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta, date
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle

# 常量定义
PEAK_HOURS = (7, 22)
VALLEY_HOURS = (0, 6)
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TARGET_COLUMNS = {
    'load': 'load',
    'pv': 'pv',
    'wind': 'wind'
}

# 2024年法定节假日日期 (基于用户提供信息)
# 注意：调休上班的日期 *不是* 节假日
HOLIDAYS_2024 = {
    # 元旦
    date(2024, 1, 1),
    # 春节 (注意：2月9日除夕建议休息，但不是法定假日)
    date(2024, 2, 10), date(2024, 2, 11), date(2024, 2, 12), date(2024, 2, 13),
    date(2024, 2, 14), date(2024, 2, 15), date(2024, 2, 16), date(2024, 2, 17),
    # 清明节
    date(2024, 4, 4), date(2024, 4, 5), date(2024, 4, 6),
    # 劳动节
    date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3), date(2024, 5, 4), date(2024, 5, 5),
    # 端午节
    date(2024, 6, 10),
    # 中秋节
    date(2024, 9, 15), date(2024, 9, 16), date(2024, 9, 17),
    # 国庆节
    date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3), date(2024, 10, 4),
    date(2024, 10, 5), date(2024, 10, 6), date(2024, 10, 7),
}

FEATURES_CONFIG = {
    'load': [
        'load_lag_1', 'load_lag_4', 'load_lag_24', 'load_lag_48', 'load_lag_168',
        'hour', 'day_of_week', 'month', 'is_holiday', 'is_peak', 'is_valley'
    ],
    'pv': [
        'pv_lag_1', 'pv_lag_4', 'pv_lag_24', 'pv_lag_48', 'pv_lag_168',
        'hour', 'day_of_week', 'month', 'is_peak'
    ],
    'wind': [
        'wind_lag_1', 'wind_lag_4', 'wind_lag_24', 'wind_lag_48', 'wind_lag_168',
        'hour', 'day_of_week', 'month'
    ]
}

# 时间特征函数
def is_peak_hour(hour, peak_hours=PEAK_HOURS):
    """判断是否为高峰时段"""
    return peak_hours[0] <= hour <= peak_hours[1]

def is_valley_hour(hour, valley_hours=VALLEY_HOURS):
    """判断是否为低谷时段"""
    return valley_hours[0] <= hour <= valley_hours[1]

def is_workday(date_obj):
    """判断是否为工作日 (考虑周末和节假日)"""
    if isinstance(date_obj, str):
        date_obj = pd.to_datetime(date_obj).date()
    elif isinstance(date_obj, pd.Timestamp):
        date_obj = date_obj.date()
        
    # 周末不是工作日
    if date_obj.weekday() >= 5:
        return False
    # 法定节假日不是工作日
    if date_obj in HOLIDAYS_2024:
        return False
    # TODO: 将调休上班日期明确标记为工作日 (如果需要，目前仅排除周末和假日)
    # 示例：调休上班日期集合
    # adjusted_workdays = {date(2024, 2, 4), date(2024, 2, 18), ...}
    # if date_obj in adjusted_workdays:
    #     return True
        
    return True

# --- 新增：创建时间特征的函数 ---
def create_temporal_features(df):
    """向DataFrame添加基于时间索引的时间特征

    Args:
        df (pd.DataFrame): 必须包含一个 DatetimeIndex。

    Returns:
        pd.DataFrame: 带有新增时间特征列的DataFrame。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame 必须包含一个 DatetimeIndex")
        
    df_with_features = df.copy()
    
    # 提取基础时间特征
    df_with_features['hour'] = df_with_features.index.hour
    df_with_features['day_of_week'] = df_with_features.index.dayofweek
    df_with_features['month'] = df_with_features.index.month
    df_with_features['day_of_year'] = df_with_features.index.dayofyear
    df_with_features['week_of_year'] = df_with_features.index.isocalendar().week.astype(int)
    df_with_features['quarter'] = df_with_features.index.quarter
    
    # 添加周期性特征 (Sin/Cos 变换)
    df_with_features['hour_sin'] = np.sin(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['hour_cos'] = np.cos(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['day_sin'] = np.sin(2 * np.pi * df_with_features['day_of_week'] / 7)
    df_with_features['day_cos'] = np.cos(2 * np.pi * df_with_features['day_of_week'] / 7)
    df_with_features['month_sin'] = np.sin(2 * np.pi * df_with_features['month'] / 12)
    df_with_features['month_cos'] = np.cos(2 * np.pi * df_with_features['month'] / 12)
    
    # 添加周末/节假日标记 (合并逻辑)
    df_dates = df_with_features.index.date
    df_with_features['is_weekend'] = (df_with_features['day_of_week'] >= 5).astype(int)
    df_with_features['is_holiday'] = np.array([d in HOLIDAYS_2024 for d in df_dates]).astype(int)
    # 工作日标记（非周末且非假日）
    df_with_features['is_workday'] = ((df_with_features['is_weekend'] == 0) & (df_with_features['is_holiday'] == 0)).astype(int)
    
    return df_with_features
# --- 时间特征函数添加结束 ---

class DataPipeline:
    """数据处理管道：处理时间序列数据，创建训练和预测数据集"""
    
    def __init__(self, forecast_type='load', seq_length=96, pred_length=96, 
                 peak_hours=PEAK_HOURS, valley_hours=VALLEY_HOURS):
        self.forecast_type = forecast_type
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.peak_hours = peak_hours
        self.valley_hours = valley_hours
        self.target_column = TARGET_COLUMNS.get(forecast_type, forecast_type)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        # 更新：动态获取特征列表，确保包含 is_holiday
        self.features = FEATURES_CONFIG.get(forecast_type, [])
        # 确保 is_holiday 在特征列表中（如果适用）
        if forecast_type == 'load' and 'is_holiday' not in self.features:
            self.features.append('is_holiday')
            
    def load_data(self, data_path):
        """加载时间序列数据并设置索引"""
        if isinstance(data_path, pd.DataFrame):
            df = data_path.copy()
        else:
            df = pd.read_csv(data_path)
        
        # 确保时间索引
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif df.index.dtype != 'datetime64[ns]':
            df.index = pd.to_datetime(df.index)
            
        return df
    
    def add_time_features(self, df):
        """添加时间特征"""
        df_with_features = df.copy()
        
        # 提取时间特征
        hours = df.index.hour
        df_dates = df.index.date # 提取日期部分
        
        df_with_features['hour'] = hours
        df_with_features['day_of_week'] = df.index.dayofweek
        df_with_features['month'] = df.index.month
        # 注意: is_workday 函数现在会考虑节假日，但这里简单按周几判断，避免循环依赖
        df_with_features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # 使用列表推导替代 apply
        df_with_features['is_peak'] = [is_peak_hour(h, self.peak_hours) for h in hours]
        df_with_features['is_valley'] = [is_valley_hour(h, self.valley_hours) for h in hours]
        
        # 添加 is_holiday 特征
        df_with_features['is_holiday'] = [d in HOLIDAYS_2024 for d in df_dates]
        # 将布尔值转换为整数 0 或 1
        df_with_features['is_holiday'] = df_with_features['is_holiday'].astype(int)
        
        return df_with_features
    
    def add_lag_features(self, df, lags=[1, 4, 24, 48, 168], column=None):
        """添加滞后特征"""
        if column is None:
            column = self.target_column
            
        df_with_lags = df.copy()
        
        for lag in lags:
            df_with_lags[f'{column}_lag_{lag}'] = df_with_lags[column].shift(lag)
            
        # 移除开始的NaN数据
        max_lag = max(lags)
        df_with_lags = df_with_lags.iloc[max_lag:]
        
        return df_with_lags
    
    def prepare_data(self, df, start_date=None, end_date=None, for_training=True):
        """准备数据集，添加特征并按需拆分训练集和测试集"""
        df_copy = df.copy()
        
        # 筛选日期范围
        if start_date and end_date:
            mask = (df_copy.index >= start_date) & (df_copy.index <= end_date)
            df_copy = df_copy.loc[mask]
            
        # 添加特征
        df_copy = self.add_time_features(df_copy)
        df_copy = self.add_lag_features(df_copy, column=self.target_column)
        
        # 处理缺失值
        df_copy = df_copy.dropna()
        
        # 分离特征和目标
        X = df_copy[self.features].values
        y = df_copy[self.target_column].values
        
        # 拆分数据集（仅用于训练）
        if for_training:
            train_size = int(len(df_copy) * TRAIN_RATIO)
            val_size = int(len(df_copy) * VALID_RATIO)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            # 保存时间戳以便于分析
            times_train = df_copy.index[:train_size]
            times_val = df_copy.index[train_size:train_size+val_size]
            times_test = df_copy.index[train_size+val_size:]
            
            # 数据缩放
            self.scaler_x.fit(X_train)
            self.scaler_y.fit(y_train.reshape(-1, 1))
            
            X_train_scaled = self.scaler_x.transform(X_train)
            X_val_scaled = self.scaler_x.transform(X_val)
            X_test_scaled = self.scaler_x.transform(X_test)
            
            y_train_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
            # 拆分为序列
            X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_scaled)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_scaled)
            X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test_scaled)
            
            # 获取峰谷标记
            is_peak_train = np.array([is_peak_hour(t.hour, self.peak_hours) for t in times_train[self.seq_length:]])
            is_peak_val = np.array([is_peak_hour(t.hour, self.peak_hours) for t in times_val[self.seq_length:]])
            is_peak_test = np.array([is_peak_hour(t.hour, self.peak_hours) for t in times_test[self.seq_length:]])
            
            is_valley_train = np.array([is_valley_hour(t.hour, self.valley_hours) for t in times_train[self.seq_length:]])
            is_valley_val = np.array([is_valley_hour(t.hour, self.valley_hours) for t in times_val[self.seq_length:]])
            is_valley_test = np.array([is_valley_hour(t.hour, self.valley_hours) for t in times_test[self.seq_length:]])
            
            # 收集序列对应的时间戳
            times_train_seq = times_train[self.seq_length:]
            times_val_seq = times_val[self.seq_length:]
            times_test_seq = times_test[self.seq_length:]
            
            return {
                'X_train': X_train_seq,
                'y_train': y_train_seq,
                'X_val': X_val_seq,
                'y_val': y_val_seq,
                'X_test': X_test_seq,
                'y_test': y_test_seq,
                'times_train': times_train_seq,
                'times_val': times_val_seq,
                'times_test': times_test_seq,
                'is_peak_train': is_peak_train,
                'is_peak_val': is_peak_val,
                'is_peak_test': is_peak_test,
                'is_valley_train': is_valley_train,
                'is_valley_val': is_valley_val,
                'is_valley_test': is_valley_test
            }
        else:
            # 用于预测的情况
            # 数据缩放
            X_scaled = self.scaler_x.transform(X)
            
            # 创建预测序列
            X_seq = []
            for i in range(len(X_scaled) - self.seq_length + 1):
                X_seq.append(X_scaled[i:i+self.seq_length])
            
            X_seq = np.array(X_seq)
            
            # 获取对应的时间戳
            timestamps = df_copy.index[self.seq_length-1:]
            
            # 获取峰谷标记
            is_peak = np.array([is_peak_hour(t.hour, self.peak_hours) for t in timestamps])
            is_valley = np.array([is_valley_hour(t.hour, self.valley_hours) for t in timestamps])
            
            return {
                'X': X_seq,
                'timestamps': timestamps,
                'is_peak': is_peak,
                'is_valley': is_valley,
                'y': y[self.seq_length-1:] if len(y) >= self.seq_length else []
            }
    
    def _create_sequences(self, X, y):
        """创建输入序列和对应的输出目标"""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i+self.seq_length])
            y_seq.append(y[i+self.seq_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def inverse_transform_y(self, y_scaled):
        """反向转换目标变量"""
        if len(y_scaled.shape) == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_scaled).flatten()
    
    def create_dataset_for_date(self, df, forecast_date, historical_days=8):
        """为指定日期创建预测数据集，使用指定天数的历史数据"""
        target_date = pd.to_datetime(forecast_date)
        start_date = target_date - timedelta(days=historical_days)
        
        # 筛选历史数据
        hist_data = df[(df.index >= start_date) & (df.index < target_date)].copy()
        
        if hist_data.empty:
            raise ValueError(f"无法找到从 {start_date} 到 {target_date} 的历史数据")
        
        return self.prepare_data(hist_data, for_training=False)
    
    def save_scalers(self, save_dir):
        """保存数据缩放器"""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'scaler_x.pkl'), 'wb') as f:
            pickle.dump(self.scaler_x, f)
            
        with open(os.path.join(save_dir, 'scaler_y.pkl'), 'wb') as f:
            pickle.dump(self.scaler_y, f)
    
    def load_scalers(self, load_dir):
        """从文件加载缩放器"""
        # 检查目录是否存在
        if not os.path.exists(load_dir):
            print(f"错误: 缩放器目录不存在 {load_dir}")
            return False
        try:
            x_path = os.path.join(load_dir, 'X_scaler.pkl')
            y_path = os.path.join(load_dir, 'y_scaler.pkl')
            
            if os.path.exists(x_path) and os.path.exists(y_path):
                # --- 使用 pickle 加载以保持一致性 ---
                with open(x_path, 'rb') as f:
                    self.scaler_x = pickle.load(f)
                with open(y_path, 'rb') as f:
                    self.scaler_y = pickle.load(f)
                # -------------------------------------
                print(f"成功加载缩放器于: {load_dir}")
                return True
            else:
                print(f"错误: 未找到缩放器文件于 {load_dir}")
                return False
        except Exception as e:
            print(f"加载缩放器时出错: {e}")
            return False

def preprocess_data_for_training(data_path, forecast_type='load', start_date=None, end_date=None, 
                               seq_length=96, pred_length=1, peak_hours=PEAK_HOURS, valley_hours=VALLEY_HOURS):
    """预处理数据用于训练，返回完整的数据字典"""
    pipeline = DataPipeline(
        forecast_type=forecast_type,
        seq_length=seq_length,
        pred_length=pred_length,
        peak_hours=peak_hours,
        valley_hours=valley_hours
    )
    
    # 加载数据
    df = pipeline.load_data(data_path)
    
    # 准备数据
    data_dict = pipeline.prepare_data(df, start_date, end_date, for_training=True)
    
    # 添加管道对象到字典中，以便后续使用
    data_dict['pipeline'] = pipeline
    
    return data_dict

class IntervalPredictionDataset(Dataset):
    """用于区间预测的PyTorch数据集"""
    
    def __init__(self, X, y=None, is_peak=None, is_valley=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
            
        if is_peak is not None:
            self.is_peak = torch.tensor(is_peak, dtype=torch.float32)
        else:
            self.is_peak = None
            
        if is_valley is not None:
            self.is_valley = torch.tensor(is_valley, dtype=torch.float32)
        else:
            self.is_valley = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = {'X': self.X[idx]}
        
        if self.y is not None:
            sample['y'] = self.y[idx]
            
        if self.is_peak is not None:
            sample['is_peak'] = self.is_peak[idx]
            
        if self.is_valley is not None:
            sample['is_valley'] = self.is_valley[idx]
            
        return sample

def create_prediction_intervals(predictions, errors, confidence_level=0.9):
    """基于预测值和历史误差创建预测区间"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
        
    if isinstance(errors, dict):
        # 使用分段误差统计
        intervals = {}
        alpha = (1 - confidence_level) / 2
        
        # 创建全部区间
        all_intervals = np.zeros((len(predictions), 2))
        
        for i, pred in enumerate(predictions):
            hour = i % 24  # 假设预测间隔是小时级
            
            # 确定时段类型
            if is_peak_hour(hour):
                error_stats = errors.get('peak', errors.get('all'))
            elif is_valley_hour(hour):
                error_stats = errors.get('valley', errors.get('all'))
            else:
                error_stats = errors.get('normal', errors.get('all'))
            
            # 计算区间
            lower_bound = pred + np.percentile(error_stats, alpha * 100)
            upper_bound = pred + np.percentile(error_stats, (1 - alpha) * 100)
            
            all_intervals[i, 0] = max(0, lower_bound)  # 确保非负
            all_intervals[i, 1] = upper_bound
        
        return all_intervals
    else:
        # 使用全局误差分布
        alpha = (1 - confidence_level) / 2
        lower_percentile = np.percentile(errors, alpha * 100)
        upper_percentile = np.percentile(errors, (1 - alpha) * 100)
        
        lower_bounds = np.maximum(0, predictions + lower_percentile)  # 确保非负
        upper_bounds = predictions + upper_percentile
        
        return np.column_stack((lower_bounds, upper_bounds))

def plot_interval_forecast(timestamps, predictions, intervals, actuals=None, title="区间预测结果"):
    """绘制区间预测结果图表"""
    plt.figure(figsize=(12, 6))
    
    # 绘制预测值
    plt.plot(timestamps, predictions, 'b-', label='预测值')
    
    # 绘制区间
    plt.fill_between(timestamps, 
                    intervals[:, 0], 
                    intervals[:, 1], 
                    color='b', alpha=0.2, 
                    label='预测区间')
    
    # 如果有实际值，也绘制出来
    if actuals is not None and len(actuals) == len(predictions):
        plt.plot(timestamps, actuals, 'r-', label='实际值')
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf() 