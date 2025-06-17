#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
天气数据处理工具使用示例

本脚本展示了如何使用utils/weather_utils.py中的工具函数来处理天气数据，
包括加载、预处理、特征工程、可视化、合并负荷数据以及创建滞后特征等功能。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys


# 添加项目根目录到路径，以便导入utils模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.plot_style

from utils.weather_utils import (
    load_weather_data,
    add_weather_engineered_features,
    plot_weather_features,
    merge_weather_and_load,
    create_lagged_weather_features,
    filter_weather_features
)

def main():
    """
    天气数据处理工具使用示例的主函数
    """
    print("=" * 80)
    print("天气数据处理工具使用示例")
    print("=" * 80)
    
    # 设置数据文件路径
    # 注意：请确保这些文件存在，或者修改为您实际的文件路径
    weather_file = 'data/timeseries_load_weather_fujian_20250608_170543.csv'
    load_file = 'data/timeseries_load_福建.csv'
    
    # 1. 加载并预处理天气数据
    print("\n1. 加载并预处理天气数据")
    print("-" * 50)
    
    try:
        weather_df = load_weather_data(
            weather_file,
            interpolate=True,              # 插值缺失值
            filter_outliers=True,          # 过滤异常值
            add_engineered_features=True   # 添加工程特征
        )
        
        # 显示数据基本信息
        print("\n天气数据基本信息:")
        print(f"形状: {weather_df.shape}")
        print(f"列: {weather_df.columns.tolist()}")
        print("\n天气数据前5行:")
        print(weather_df.head())
        
    except Exception as e:
        print(f"加载天气数据时出错: {str(e)}")
        print(f"请确保文件 '{weather_file}' 存在，或修改为您实际的文件路径")
        # 创建一个示例天气数据帧用于演示
        print("\n创建示例天气数据用于演示...")
        weather_df = create_sample_weather_data()
    
    # 2. 天气特征工程（如果在load_weather_data中没有添加）
    print("\n2. 天气特征工程")
    print("-" * 50)
    
    # 添加工程特征（如果在load_weather_data中设置了add_engineered_features=True，则可以跳过此步骤）
    weather_df = add_weather_engineered_features(weather_df)
    
    # 显示添加的特征
    original_cols = ['timestamp', 'temperature', 'humidity', 'wind_speed', 'precipitation']
    engineered_cols = [col for col in weather_df.columns if col not in original_cols]
    
    print(f"添加的工程特征: {engineered_cols}")
    print("\n工程特征数据示例:")
    print(weather_df[['timestamp'] + engineered_cols[:3]].head())  # 只显示前3个工程特征
    
    # 3. 天气数据可视化
    print("\n3. 天气数据可视化")
    print("-" * 50)
    
    # 获取数据的时间范围
    min_date = weather_df['timestamp'].min()
    max_date = weather_df['timestamp'].max()
    
    # 选择一个7天的时间窗口进行可视化
    start_date = min_date + (max_date - min_date) * 0.2  # 从20%位置开始
    end_date = start_date + timedelta(days=7)  # 7天窗口
    
    print(f"可视化时间范围: {start_date.date()} 到 {end_date.date()}")
    
    # 选择要可视化的特征
    viz_features = ['temperature', 'humidity', 'wind_speed']
    if 'heat_index' in weather_df.columns:
        viz_features.append('heat_index')
    if 'feels_like' in weather_df.columns:
        viz_features.append('feels_like')
    
    # 绘制天气特征
    try:
        plot_weather_features(
            weather_df,
            features=viz_features,
            start_date=start_date,
            end_date=end_date,
            figsize=(15, 10)
        )
        print("已绘制天气特征图表。")
    except Exception as e:
        print(f"绘制天气特征时出错: {str(e)}")
    
    # 4. 加载负荷数据并与天气数据合并
    print("\n4. 加载负荷数据并与天气数据合并")
    print("-" * 50)
    
    try:
        # 加载负荷数据
        load_df = pd.read_csv(load_file)
        load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
        
        print(f"负荷数据形状: {load_df.shape}")
        print("负荷数据前5行:")
        print(load_df.head())
        
        # 合并负荷和天气数据
        merged_df = merge_weather_and_load(
            load_df,
            weather_df,
            interpolate_weather=True  # 对天气数据进行插值以匹配所有负荷时间戳
        )
        
        print(f"\n合并后数据形状: {merged_df.shape}")
        print("合并后数据前5行:")
        print(merged_df.head())
        
    except Exception as e:
        print(f"处理负荷数据时出错: {str(e)}")
        print(f"请确保文件 '{load_file}' 存在，或修改为您实际的文件路径")
        # 创建一个示例合并数据帧用于演示
        print("\n创建示例合并数据用于演示...")
        merged_df = create_sample_merged_data(weather_df)
    
    # 5. 创建滞后天气特征
    print("\n5. 创建滞后天气特征")
    print("-" * 50)
    
    # 选择要创建滞后特征的天气特征
    lag_features = ['temperature', 'humidity', 'wind_speed']
    if 'heat_index' in merged_df.columns:
        lag_features.append('heat_index')
    if 'feels_like' in merged_df.columns:
        lag_features.append('feels_like')
    
    # 设置滞后小时数
    lag_hours = [1, 3, 6, 24]
    
    print(f"为以下特征创建滞后版本: {lag_features}")
    print(f"滞后小时数: {lag_hours}")
    
    # 创建滞后特征
    df_with_lags = create_lagged_weather_features(
        merged_df,
        weather_features=lag_features,
        lag_hours=lag_hours
    )
    
    print(f"\n添加滞后特征后的数据形状: {df_with_lags.shape}")
    
    # 计算添加了多少个滞后特征
    n_lag_features = len(df_with_lags.columns) - len(merged_df.columns)
    print(f"添加了 {n_lag_features} 个滞后特征")
    
    # 显示部分滞后特征
    lag_cols = [col for col in df_with_lags.columns if '_lag' in col]
    print(f"\n滞后特征示例 (前5个): {lag_cols[:5]}")
    print(df_with_lags[['timestamp', 'load'] + lag_cols[:3]].head())  # 只显示前3个滞后特征
    
    # 6. 基于相关性筛选天气特征
    print("\n6. 基于相关性筛选天气特征")
    print("-" * 50)
    
    if 'load' in df_with_lags.columns:
        # 筛选相关性高的特征
        selected_features = filter_weather_features(
            df_with_lags,
            correlation_threshold=0.1,  # 最小相关性阈值
            max_features=10,            # 保留的最大特征数
            target_col='load'           # 目标列名
        )
        
        print(f"\n选择了 {len(selected_features)} 个特征:")
        print(selected_features)
        
        # 7. 创建最终的训练数据集
        print("\n7. 创建最终的训练数据集")
        print("-" * 50)
        
        X = df_with_lags[selected_features]
        y = df_with_lags['load']
        
        print(f"最终数据集形状: X={X.shape}, y={y.shape}")
        
        # 显示最终数据集的前几行
        print("\n最终特征数据集前5行:")
        print(X.head())
    else:
        print("数据中没有'load'列，无法进行特征筛选和创建训练数据集。")
    
    print("\n" + "=" * 80)
    print("天气数据处理示例完成")
    print("=" * 80)

def create_sample_weather_data(n_samples=1000):
    """
    创建示例天气数据用于演示
    
    Args:
        n_samples: 样本数量
        
    Returns:
        pandas.DataFrame: 示例天气数据
    """
    # 创建时间戳
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # 创建基本天气数据
    import numpy as np
    
    # 温度: 10-30摄氏度，有日变化和随机波动
    hour_of_day = np.array([t.hour for t in timestamps])
    day_of_year = np.array([(t - datetime(t.year, 1, 1)).days for t in timestamps])
    
    # 温度有日变化和季节变化
    temperature = 20 + 5 * np.sin(2 * np.pi * day_of_year / 365) + 5 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 1, n_samples)
    
    # 湿度: 30-90%，与温度负相关
    humidity = 60 - 0.5 * (temperature - 20) + np.random.normal(0, 5, n_samples)
    humidity = np.clip(humidity, 30, 90)
    
    # 风速: 0-10 m/s，随机
    wind_speed = np.abs(np.random.normal(3, 2, n_samples))
    wind_speed = np.clip(wind_speed, 0, 10)
    
    # 降水: 大部分为0，偶尔有降水
    precipitation = np.zeros(n_samples)
    rain_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)  # 10%的时间有雨
    precipitation[rain_idx] = np.abs(np.random.exponential(1, size=len(rain_idx)))
    
    # 创建数据帧
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'precipitation': precipitation
    })
    
    return df

def create_sample_merged_data(weather_df, n_samples=1000):
    """
    创建示例合并数据用于演示
    
    Args:
        weather_df: 天气数据帧
        n_samples: 样本数量
        
    Returns:
        pandas.DataFrame: 示例合并数据
    """
    # 如果有实际的天气数据，使用相同的时间戳
    if len(weather_df) > 0:
        timestamps = weather_df['timestamp'].values[:min(n_samples, len(weather_df))]
        n_samples = len(timestamps)
    else:
        # 创建时间戳
        start_date = datetime(2023, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # 创建负荷数据，负荷与时间和温度相关
    import numpy as np
    
    # 获取小时和工作日信息
    hour_of_day = np.array([t.hour for t in timestamps])
    is_weekend = np.array([t.weekday() >= 5 for t in timestamps]).astype(int)
    
    # 基础负荷
    base_load = 1000
    
    # 时间模式：工作日9-17点高峰，周末较低
    time_pattern = 300 * np.sin(np.pi * (hour_of_day - 5) / 12) * (hour_of_day >= 5) * (hour_of_day <= 17)
    time_pattern = time_pattern * (1 - 0.3 * is_weekend)
    
    # 如果有温度数据，使用它；否则创建模拟温度
    if 'temperature' in weather_df.columns and len(weather_df) > 0:
        temperature = weather_df['temperature'].values[:n_samples]
    else:
        temperature = 20 + 5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 30)) + np.random.normal(0, 1, n_samples)
    
    # 温度影响：温度过高或过低时负荷增加（空调和暖气）
    temp_effect = 100 * ((temperature - 22)**2 / 40)
    
    # 随机波动
    random_effect = np.random.normal(0, 50, n_samples)
    
    # 总负荷
    load = base_load + time_pattern + temp_effect + random_effect
    load = np.clip(load, 500, 2000)  # 限制在合理范围内
    
    # 创建合并数据帧
    if len(weather_df) > 0 and n_samples <= len(weather_df):
        # 使用实际天气数据
        merged_df = weather_df.iloc[:n_samples].copy()
        merged_df['load'] = load
    else:
        # 创建模拟数据
        merged_df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'humidity': np.random.uniform(30, 90, n_samples),
            'wind_speed': np.random.uniform(0, 10, n_samples),
            'precipitation': np.zeros(n_samples),
            'load': load
        })
    
    return merged_df

if __name__ == "__main__":
    main() 