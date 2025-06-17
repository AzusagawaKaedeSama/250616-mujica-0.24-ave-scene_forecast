import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

def identify_peak_hours(df, peak_start=8, peak_end=20, peak_days=(0, 1, 2, 3, 4)):
    """
    识别峰值时段
    
    Args:
        df: 包含timestamp列的数据帧
        peak_start: 峰值开始小时 (24小时制)
        peak_end: 峰值结束小时 (24小时制)
        peak_days: 工作日列表 (0=周一, 6=周日)
        
    Returns:
        numpy.ndarray: 布尔数组，表示每个时间戳是否属于峰值时段
    """
    if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 提取小时和星期几
    hours = df['timestamp'].dt.hour
    days_of_week = df['timestamp'].dt.dayofweek
    
    # 判断是否是峰值时段
    is_peak_hour = (hours >= peak_start) & (hours < peak_end)
    is_peak_day = days_of_week.isin(peak_days)
    
    # 只有当同时满足峰值小时和工作日时，才被认为是峰值时段
    is_peak = is_peak_hour & is_peak_day
    
    return is_peak

def calculate_distance_to_peak(df, is_peak):
    """
    计算每个时间点到最近峰值时段的距离（小时）
    
    Args:
        df: 包含timestamp列的数据帧
        is_peak: 布尔数组，表示每个时间戳是否属于峰值时段
        
    Returns:
        numpy.ndarray: 到最近峰值时段的距离（小时）
    """
    if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 获取时间戳数组
    timestamps = df['timestamp'].to_numpy()
    
    # 初始化距离数组
    distance = np.zeros(len(timestamps))
    
    # 对于非峰值时段的每个时间点
    for i in range(len(timestamps)):
        if not is_peak[i]:
            # 找出所有峰值时段的索引
            peak_indices = np.where(is_peak)[0]
            if len(peak_indices) == 0:
                distance[i] = 24  # 如果没有峰值时段，设置默认值
                continue
                
            # 计算到所有峰值时段的时间差（小时）
            time_diffs = np.array([
                abs((timestamps[i] - timestamps[j]).total_seconds() / 3600)
                for j in peak_indices
            ])
            
            # 找出最小距离
            distance[i] = np.min(time_diffs)
    
    return distance

def estimate_peak_magnitude(df, load_col='load', window_size=24*7):
    """
    估计峰值幅度（相对于移动平均的偏差）
    
    Args:
        df: 包含负荷列的数据帧
        load_col: 负荷列名
        window_size: 移动平均窗口大小
        
    Returns:
        numpy.ndarray: 峰值幅度（相对于移动平均的偏差）
    """
    # 计算移动平均
    if load_col in df.columns:
        load = df[load_col].values
        # 使用中心化的移动平均
        moving_avg = pd.Series(load).rolling(window=window_size, center=True, min_periods=1).mean().values
        
        # 计算相对偏差
        magnitude = (load - moving_avg) / (moving_avg + 1e-8)  # 避免除零错误
        
        # 将偏差限制在合理范围内
        magnitude = np.clip(magnitude, -1.0, 1.0)
        
        return magnitude
    else:
        print(f"警告: 找不到列 '{load_col}'，返回零数组")
        return np.zeros(len(df))

def add_peak_awareness_features(df, peak_start=8, peak_end=20, peak_days=(0, 1, 2, 3, 4), load_col='load'):
    """
    向数据帧添加峰值感知特征
    
    Args:
        df: 包含timestamp和load列的数据帧
        peak_start: 峰值开始小时
        peak_end: 峰值结束小时
        peak_days: 工作日列表
        load_col: 负荷列名
        
    Returns:
        tuple: (带峰值特征的数据帧, 峰值信息字典)
    """
    # 复制数据帧，避免修改原始数据
    df = df.copy()
    
    # 确保timestamp列是datetime类型
    if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 识别峰值时段
    is_peak = identify_peak_hours(df, peak_start, peak_end, peak_days)
    df['is_peak'] = is_peak.astype(int)  # 转换为0/1
    
    # 计算到最近峰值的距离
    distance_to_peak = calculate_distance_to_peak(df, is_peak)
    df['distance_to_peak'] = distance_to_peak
    
    # 估计峰值幅度
    peak_magnitude = estimate_peak_magnitude(df, load_col)
    df['peak_magnitude'] = peak_magnitude
    
    # 收集峰值信息
    peak_info = {
        'peak_hours': (peak_start, peak_end),
        'peak_days': peak_days,
        'peak_percentage': is_peak.mean() * 100,
        'avg_peak_magnitude': peak_magnitude[is_peak].mean() if is_peak.sum() > 0 else 0
    }
    
    return df, peak_info

def extract_cyclical_time_features(df):
    """
    提取时间的周期性特征
    
    Args:
        df: 包含timestamp列的数据帧
        
    Returns:
        pandas.DataFrame: 带有周期性时间特征的数据帧
    """
    # 复制数据帧，避免修改原始数据
    df = df.copy()
    
    # 确保timestamp列是datetime类型
    if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 提取小时特征（正弦和余弦编码）
    hour = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # 提取星期几特征（正弦和余弦编码）
    day_of_week = df['timestamp'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # 提取月份特征（正弦和余弦编码）
    month = df['timestamp'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    return df

def create_peak_aware_sequences(df, feature_cols, target_col='load', seq_length=96, pred_length=4):
    """
    创建考虑峰值特征的序列数据
    
    Args:
        df: 带有峰值特征的数据帧
        feature_cols: 特征列名列表
        target_col: 目标列名
        seq_length: 输入序列长度
        pred_length: 预测长度
        
    Returns:
        tuple: (X, y, is_peak_sequence, timestamps)
    """
    # 确保所有需要的列都存在
    for col in feature_cols + [target_col, 'is_peak']:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不在数据帧中")
    
    # 提取特征和目标数据
    X = []
    y = []
    is_peak = []
    timestamps = []
    
    # 创建序列
    for i in range(len(df) - seq_length - pred_length + 1):
        # 输入序列
        features = df[feature_cols].iloc[i:i+seq_length].values
        X.append(features)
        
        # 目标序列
        target = df[target_col].iloc[i+seq_length:i+seq_length+pred_length].values
        y.append(target)
        
        # 最后一个输入时刻是否为峰值
        last_is_peak = df['is_peak'].iloc[i+seq_length-1]
        is_peak.append(last_is_peak)
        
        # 记录预测起始时间戳
        timestamps.append(df['timestamp'].iloc[i+seq_length])
    
    return np.array(X), np.array(y), np.array(is_peak), np.array(timestamps)

def create_holiday_feature(df, country='US'):
    """
    添加节假日特征
    
    Args:
        df: 包含timestamp列的数据帧
        country: 国家代码，用于获取对应国家的节假日
        
    Returns:
        pandas.DataFrame: 带有节假日特征的数据帧
    """
    try:
        import holidays
    except ImportError:
        print("缺少holidays库，请安装: pip install holidays")
        # 如果没有安装holidays库，返回全0的特征
        df['is_holiday'] = 0
        return df
        
    # 复制数据帧，避免修改原始数据
    df = df.copy()
    
    # 确保timestamp列是datetime类型
    if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 获取对应国家的节假日
    try:
        holiday_dict = holidays.country_holidays(country, years=range(
            df['timestamp'].min().year, 
            df['timestamp'].max().year + 1
        ))
    except:
        print(f"警告: 无法加载国家 '{country}' 的节假日，返回全0的特征")
        df['is_holiday'] = 0
        return df
    
    # 检查每个时间是否是节假日
    df['is_holiday'] = df['timestamp'].dt.date.map(
        lambda x: 1 if x in holiday_dict else 0
    )
    
    # 添加前一天和后一天是否是节假日的特征
    dates = df['timestamp'].dt.date.unique()
    next_day_dict = {d: d + timedelta(days=1) in holiday_dict for d in dates}
    prev_day_dict = {d: d - timedelta(days=1) in holiday_dict for d in dates}
    
    df['next_day_holiday'] = df['timestamp'].dt.date.map(
        lambda x: 1 if next_day_dict.get(x, False) else 0
    )
    
    df['prev_day_holiday'] = df['timestamp'].dt.date.map(
        lambda x: 1 if prev_day_dict.get(x, False) else 0
    )
    
    return df 