"""
数据集工具模块 - 提供健壮的数据准备和处理功能
适用于天气增强型负荷预测模型的数据准备

此模块基于现有的DatasetBuilder优化实现，增加了更多数据验证和错误处理。
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from .scaler_manager import ScalerManager

# 尝试导入峰值工具模块
try:
    from utils.peak_utils import add_peak_awareness_features as add_peaks
    PEAK_UTILS_AVAILABLE = True
except ImportError:
    PEAK_UTILS_AVAILABLE = False
    print("警告: 未找到峰值处理工具模块，将不使用峰值特征")

def prepare_timeseries_data(
    data_file, 
    seq_length, 
    pred_length,
    include_peaks=False,
    train_start_date=None,
    train_end_date=None,
    debug=False):
    """
    一个健壮的数据准备函数，用于加载、处理和序列化时间序列数据
    
    Args:
        data_file (str): CSV文件路径
        seq_length (int): 输入序列长度
        pred_length (int): 预测序列长度
        include_peaks (bool): 是否添加峰值感知特征
        train_start_date (str, optional): 训练数据开始日期 (YYYY-MM-DD)
        train_end_date (str, optional): 训练数据结束日期 (YYYY-MM-DD)
        debug (bool): 是否打印调试信息
        
    Returns:
        tuple: 包含训练/验证集 (X_train, y_train, X_val, y_val), 
               峰值信息 (train_is_peak, val_is_peak),
               以及维度信息 (input_dim, weather_dim)
    """
    
    # 1. 加载数据
    try:
        df = pd.read_csv(data_file)
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("数据文件中必须包含 'datetime' 或 'timestamp' 列")
        
        df = df.set_index('timestamp').sort_index()
    except Exception as e:
        raise IOError(f"加载或解析数据文件 '{data_file}' 时出错: {e}")
        
    # 根据日期范围筛选数据
    if train_start_date and train_end_date:
        if debug:
            print(f"根据日期筛选数据: 从 {train_start_date} 到 {train_end_date}")
        
        start_date = pd.to_datetime(train_start_date)
        end_date = pd.to_datetime(train_end_date)
        
        # 确保时间范围有效
        if start_date >= end_date:
            raise ValueError("训练开始日期必须在结束日期之前")
            
        df = df.loc[start_date:end_date]
        
        if df.empty:
            raise ValueError(f"在指定的日期范围 {train_start_date} - {train_end_date} 内没有找到数据")

    # 2. 数据清洗和验证
    if 'load' not in df.columns:
        raise ValueError("数据文件中必须包含'load'列")
    
    # 确保数据包含负荷列
    load_col = ensure_load_column(df, debug)
    
    # 处理时间戳
    ensure_timestamp_column(df, debug)
    
    # 去除非数值列
    non_numeric_cols = [col for col in df.columns 
                       if (df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])) 
                       and col != 'timestamp']
    
    if non_numeric_cols and debug:
        print(f"去除非数值列: {non_numeric_cols}")
        
    df = df.drop(columns=non_numeric_cols)
    
    # 添加时间特征
    df = add_time_features(df, debug)
    
    # 识别特征类型
    time_features = ['hour', 'day_of_week', 'month']
    weather_features = identify_weather_features(df, load_col, time_features, debug)
    
    # 添加峰值感知特征
    train_is_peak = None
    val_is_peak = None
    
    if include_peaks and PEAK_UTILS_AVAILABLE:
        df, peak_info = add_peak_awareness_features(df, load_col, debug)
        if debug and 'is_peak' in df.columns:
            print(f"峰值样本比例: {df['is_peak'].mean():.2%}")
    
    # 处理缺失值
    handle_missing_values(df, debug)
    
    # 确保有足够数据
    if len(df) < seq_length + pred_length:
        raise ValueError(f"数据长度 ({len(df)}) 小于所需的序列长度+预测长度 ({seq_length + pred_length})")
    
    # 准备特征和目标列
    feature_cols = time_features + weather_features
    peak_cols = [col for col in df.columns if col.startswith('is_peak')]
    if include_peaks and peak_cols:
        feature_cols += peak_cols
    
    if debug:
        print(f"特征列: {feature_cols}")
        print(f"目标列: {load_col}")
    
    # 创建序列
    try:
        X, y, timestamps, is_peak = create_sequences(df, feature_cols, load_col, seq_length, pred_length)
        
        if debug:
            print(f"序列形状 - X: {X.shape}, y: {y.shape}")
    except Exception as e:
        print(f"创建序列时出错: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 划分数据集
    train_size = int(len(X) * 0.8)
    val_size = len(X) - train_size
    
    if debug:
        print(f"数据集划分 - 训练集: {train_size}, 验证集: {val_size}, 剩余: {len(X) - train_size - val_size}")
    
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    train_timestamps, val_timestamps = timestamps[:train_size], timestamps[train_size:]
    
    # 提取峰值信息（如果有）
    if include_peaks and peak_cols:
        peak_idx = feature_cols.index(peak_cols[0]) if peak_cols else -1
        
        if peak_idx >= 0:
            # 从最后一个时间步提取峰值信息
            train_is_peak = X_train[:, -1, peak_idx].copy().astype(bool)
            val_is_peak = X_val[:, -1, peak_idx].copy().astype(bool)
            
            if debug:
                print(f"训练集峰值样本比例: {train_is_peak.mean():.2%}")
                print(f"验证集峰值样本比例: {val_is_peak.mean() if len(val_is_peak) > 0 else 0:.2%}")
    
    # 检查数据有效性
    validate_data(X_train, y_train, X_val, y_val, debug)
    
    # 计算特征维度
    input_dim = len(time_features)
    weather_dim = len(weather_features)
    
    if debug:
        print(f"特征维度 - 时间: {input_dim}, 天气: {weather_dim}, 总计: {input_dim + weather_dim}")
    
    # 返回拆分和处理后的数据，但不进行归一化
    return (
        X_train, y_train, X_val, y_val, 
        train_is_peak, val_is_peak,
        input_dim, weather_dim
    )

def ensure_load_column(df, debug=False):
    """
    确保数据包含负荷列
    
    Args:
        df: 数据帧
        debug: 是否打印调试信息
        
    Returns:
        str: 负荷列名
    """
    # 检查是否有'load'列
    if 'load' in df.columns:
        if debug:
            print("找到负荷列: 'load'")
        return 'load'
    
    # 检查可能的负荷列名
    load_candidates = [
        col for col in df.columns if any(term in col.lower() for term in [
            'load', 'power', 'electricity', 'consumption', 
            '负荷', '电量', '功率', '用电'
        ])
    ]
    
    if load_candidates:
        load_col = load_candidates[0]
        if debug:
            print(f"使用'{load_col}'作为负荷列")
        # 创建标准名称的副本
        df['load'] = df[load_col].copy()
        return 'load'
    
    # 如果找不到，尝试使用第二列（假设第一列是时间戳或索引）
    if len(df.columns) > 1:
        potential_load_col = df.columns[1]
        print(f"警告: 未找到明确的负荷列，尝试使用'{potential_load_col}'作为负荷列")
        df['load'] = df[potential_load_col].copy()
        return 'load'
    
    raise ValueError("无法确定负荷数据列。请确保数据包含'load'列或类似名称的列。")

def ensure_timestamp_column(df, debug=False):
    """
    确保数据包含时间戳列，如果没有则创建
    
    Args:
        df: 数据帧
        debug: 是否打印调试信息
        
    Returns:
        None: 直接修改数据帧
    """
    # 检查是否有'timestamp'列
    timestamp_col = 'timestamp'
    if timestamp_col in df.columns:
        if debug:
            print("找到时间戳列: 'timestamp'")
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            return
        except Exception as e:
            if debug:
                print(f"转换时间戳列时出错: {e}")
    
    # 尝试查找可能的时间戳列
    timestamp_candidates = [
        col for col in df.columns if any(term in col.lower() for term in [
            'time', 'date', 'datetime', 'timestamp', 'period',
            '时间', '日期', '时刻'
        ])
    ]
    
    if timestamp_candidates:
        time_col = timestamp_candidates[0]
        if debug:
            print(f"使用'{time_col}'作为时间戳列")
        try:
            df[timestamp_col] = pd.to_datetime(df[time_col])
            return
        except Exception as e:
            if debug:
                print(f"转换候选时间戳列时出错: {e}")
    
    # 如果找不到，创建假设的时间序列
    print("警告: 未找到有效时间戳列，创建假设的时间序列")
    df[timestamp_col] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')

def add_time_features(df, debug=False):
    """
    从时间戳添加时间特征
    
    Args:
        df: 包含timestamp列的数据帧
        debug: 是否打印调试信息
        
    Returns:
        pandas.DataFrame: 添加了时间特征的数据帧
    """
    if 'timestamp' not in df.columns:
        raise ValueError("未找到'timestamp'列，请先确保数据包含时间戳")
    
    # 添加时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # 可选的额外时间特征
    df['day'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'] >= 5
    
    if debug:
        print("已添加时间特征: hour, day_of_week, month, day, is_weekend")
    
    return df

def identify_weather_features(df, load_col, time_features, debug=False):
    """
    识别数据中的天气特征
    
    Args:
        df: 数据帧
        load_col: 负荷列名
        time_features: 时间特征列表
        debug: 是否打印调试信息
        
    Returns:
        list: 天气特征列名列表
    """
    exclude_cols = ['timestamp', load_col, 'index', 'id'] + time_features
    exclude_cols += [col for col in df.columns if col.startswith('is_peak')]
    
    # 假设剩余的数值列是天气特征
    weather_features = [col for col in df.columns 
                       if col not in exclude_cols 
                       and is_numeric_column(df[col])]
    
    if not weather_features and debug:
        print("警告: 未识别出天气特征，模型性能可能会受到影响")
    elif debug:
        print(f"识别出的天气特征 ({len(weather_features)}): {weather_features}")
    
    return weather_features

def is_numeric_column(series):
    """检查列是否为数值类型"""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

def add_peak_awareness_features(df, load_col='load', debug=False):
    """
    添加峰值感知特征
    
    Args:
        df: 数据帧
        load_col: 负荷列名
        debug: 是否打印调试信息
        
    Returns:
        tuple: (更新的数据帧, 峰值信息)
    """
    if not PEAK_UTILS_AVAILABLE:
        if 'hour' not in df.columns:
            raise ValueError("添加峰值特征需要'hour'列")
        
        # 简单峰值检测（电力负荷通常在9-11点和18-20点达到峰值）
        df['is_peak'] = ((df['hour'] >= 9) & (df['hour'] <= 11)) | ((df['hour'] >= 18) & (df['hour'] <= 20))
        
        if debug:
            print("使用简单时段定义的峰值")
        
        # 创建峰值信息字典
        peak_info = {
            'method': 'time_based',
            'peak_hours': [9, 10, 11, 18, 19, 20],
            'peak_count': df['is_peak'].sum(),
            'peak_ratio': df['is_peak'].mean()
        }
    else:
        # 使用峰值工具模块
        try:
            df, peak_info = add_peaks(df)
            if debug:
                print("使用峰值工具模块添加峰值特征")
        except Exception as e:
            print(f"使用峰值工具添加峰值特征时出错: {e}")
            # 回退到简单方法
            df['is_peak'] = ((df['hour'] >= 9) & (df['hour'] <= 11)) | ((df['hour'] >= 18) & (df['hour'] <= 20))
            peak_info = {
                'method': 'time_based_fallback',
                'peak_hours': [9, 10, 11, 18, 19, 20],
                'peak_count': df['is_peak'].sum(),
                'peak_ratio': df['is_peak'].mean()
            }
            if debug:
                print("回退到使用简单时段定义的峰值")
    
    return df, peak_info

def handle_missing_values(df, debug=False):
    """
    处理数据中的缺失值
    
    Args:
        df: 数据帧
        debug: 是否打印调试信息
        
    Returns:
        pandas.DataFrame: 处理了缺失值的数据帧
    """
    # 检查缺失值
    missing_count = df.isna().sum()
    missing_total = missing_count.sum()
    
    if missing_total > 0:
        if debug:
            print(f"发现{missing_total}个缺失值:")
            print(missing_count[missing_count > 0])
        
        # 先用前向填充，再用后向填充
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        # 检查是否还有缺失值
        remaining_missing = df.isna().sum().sum()
        if remaining_missing > 0:
            print(f"警告: 填充后仍有{remaining_missing}个缺失值，将用列均值填充")
            for col in df.columns:
                if df[col].isna().sum() > 0:
                    df[col].fillna(df[col].mean(), inplace=True)
    
    return df

def create_sequences(df, feature_cols, target_col, seq_length, pred_length):
    """
    从数据帧创建输入序列和目标序列
    
    Args:
        df: 数据帧
        feature_cols: 特征列名
        target_col: 目标列名
        seq_length: 输入序列长度
        pred_length: 预测长度
        
    Returns:
        tuple: (X, y, timestamps, is_peak) - 特征序列，目标序列，时间戳序列，峰值序列
    """
    if np.isnan(df[feature_cols]).any().any() or np.isnan(df[target_col]).any():
        raise ValueError("输入数据中包含NaN值，请在调用create_sequences之前处理")

    X, y, timestamps, is_peak = [], [], [], []
    for i in range(len(df) - seq_length - pred_length + 1):
        X.append(df[feature_cols].iloc[i : i + seq_length].values)
        y.append(df[target_col].iloc[i + seq_length : i + seq_length + pred_length].values)
        
        # 记录每个序列结束点的时间戳，用于后续分析
        if 'timestamp' in df.columns:
            timestamps.append(df['timestamp'].iloc[i+seq_length-1])
        
        # 记录与y对应的峰值状态
        if 'is_peak' in df.columns:
            # 检查整个预测窗口是否包含任何峰值
            is_peak_window = df['is_peak'].iloc[i + seq_length : i + seq_length + pred_length].any()
            is_peak.append(is_peak_window)

    X = np.array(X)
    y = np.array(y)
    timestamps = np.array(timestamps)
    is_peak = np.array(is_peak) if is_peak else None
    
    return X, y, timestamps, is_peak

def validate_data(X_train, y_train, X_val, y_val, debug=False):
    """
    验证数据集的有效性
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_val: 验证特征
        y_val: 验证目标
        debug: 是否打印调试信息
        
    Returns:
        bool: 数据是否有效
    """
    # 检查是否有NaN值
    if np.isnan(X_train).any():
        print(f"警告: 训练特征中包含{np.isnan(X_train).sum()}个NaN值")
    
    if np.isnan(y_train).any():
        print(f"警告: 训练目标中包含{np.isnan(y_train).sum()}个NaN值")
    
    if X_val is not None and np.isnan(X_val).any():
        print(f"警告: 验证特征中包含{np.isnan(X_val).sum()}个NaN值")
    
    if y_val is not None and np.isnan(y_val).any():
        print(f"警告: 验证目标中包含{np.isnan(y_val).sum()}个NaN值")
    
    # 检查训练集大小
    if len(X_train) < 10:
        print(f"警告: 训练样本数量过少 ({len(X_train)})")
    
    # 检查维度匹配
    if X_train.shape[0] != y_train.shape[0]:
        print(f"警告: 训练样本数不匹配 - X: {X_train.shape[0]}, y: {y_train.shape[0]}")
    
    if X_val is not None and y_val is not None and X_val.shape[0] != y_val.shape[0]:
        print(f"警告: 验证样本数不匹配 - X: {X_val.shape[0]}, y: {y_val.shape[0]}")
    
    if debug:
        print("数据验证完成")
    
    return True 