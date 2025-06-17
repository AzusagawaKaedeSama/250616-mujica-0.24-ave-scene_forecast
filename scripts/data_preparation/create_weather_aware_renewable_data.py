#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
创建天气感知的光伏和风电数据文件
将现有的光伏/风电数据与天气数据合并，用于天气感知预测
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def create_weather_aware_renewable_data():
    """
    为光伏和风电创建包含天气数据的文件
    """
    print("=== 创建天气感知的光伏和风电数据文件 ===")
    
    data_dir = "data"
    provinces = ['上海', '安徽', '浙江', '江苏', '福建']
    renewable_types = ['pv', 'wind']
    
    for province in provinces:
        print(f"\n--- 处理省份: {province} ---")
        
        # 加载天气数据（从负荷天气文件中提取天气特征）
        weather_load_file = os.path.join(data_dir, f"timeseries_load_weather_{province}.csv")
        
        if not os.path.exists(weather_load_file):
            print(f"警告: 天气数据文件不存在: {weather_load_file}")
            continue
            
        print(f"加载天气数据: {weather_load_file}")
        weather_df = pd.read_csv(weather_load_file)
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
        weather_df = weather_df.set_index('datetime')
        
        # 提取天气特征列（排除负荷列）
        weather_columns = [col for col in weather_df.columns 
                          if col not in ['PARTY_ID', 'load'] and 'weather' in col.lower() or col in ['u10', 'v10']]
        
        print(f"天气特征列: {weather_columns}")
        
        for renewable_type in renewable_types:
            renewable_file = os.path.join(data_dir, f"timeseries_{renewable_type}_{province}.csv")
            
            if not os.path.exists(renewable_file):
                print(f"警告: 新能源数据文件不存在: {renewable_file}")
                continue
                
            print(f"处理 {renewable_type} 数据: {renewable_file}")
            
            # 加载新能源数据
            renewable_df = pd.read_csv(renewable_file)
            renewable_df['datetime'] = pd.to_datetime(renewable_df['datetime'])
            renewable_df = renewable_df.set_index('datetime')
            
            # 合并天气数据和新能源数据
            merged_df = renewable_df.join(weather_df[weather_columns], how='inner')
            
            # 重置索引以保持datetime列
            merged_df = merged_df.reset_index()
            
            # 检查合并结果
            print(f"合并前 {renewable_type} 数据行数: {len(renewable_df)}")
            print(f"合并后数据行数: {len(merged_df)}")
            print(f"天气特征数量: {len(weather_columns)}")
            
            # 保存合并后的数据
            output_file = os.path.join(data_dir, f"timeseries_{renewable_type}_weather_{province}.csv")
            merged_df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"已保存: {output_file}")
            print(f"数据列: {list(merged_df.columns)}")

def analyze_weather_features_for_renewables():
    """
    分析天气特征对光伏和风电的相关性
    """
    print("\n=== 分析天气特征对新能源预测的重要性 ===")
    
    # 光伏发电相关的天气特征
    pv_important_features = [
        'weather_temperature_c',           # 温度 - 影响光伏板效率
        'weather_relative_humidity',       # 湿度 - 影响大气透明度
        'weather_precipitation_mm',        # 降水 - 影响太阳辐射
        'weather_dewpoint_c',             # 露点温度 - 影响云量
        'u10', 'v10'                      # 风速分量 - 影响云层移动
    ]
    
    # 风电发电相关的天气特征
    wind_important_features = [
        'weather_wind_speed',             # 风速 - 直接影响风电出力
        'weather_temperature_c',          # 温度 - 影响空气密度
        'weather_relative_humidity',      # 湿度 - 影响空气密度
        'u10', 'v10',                    # 风速分量 - 风向和风速
        'weather_precipitation_mm'        # 降水 - 影响风况
    ]
    
    print("光伏发电重要天气特征:")
    for i, feature in enumerate(pv_important_features, 1):
        print(f"  {i}. {feature}")
    
    print("\n风电发电重要天气特征:")
    for i, feature in enumerate(wind_important_features, 1):
        print(f"  {i}. {feature}")
    
    return pv_important_features, wind_important_features

def validate_created_files():
    """
    验证创建的天气感知新能源数据文件
    """
    print("\n=== 验证创建的数据文件 ===")
    
    data_dir = "data"
    provinces = ['上海', '安徽', '浙江', '江苏', '福建']
    renewable_types = ['pv', 'wind']
    
    for province in provinces:
        for renewable_type in renewable_types:
            file_path = os.path.join(data_dir, f"timeseries_{renewable_type}_weather_{province}.csv")
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, nrows=5)  # 只读取前5行
                print(f"\n✅ {file_path}")
                print(f"   数据维度: {df.shape}")
                print(f"   列名: {list(df.columns)}")
                
                # 检查是否包含天气特征
                weather_cols = [col for col in df.columns if 'weather' in col.lower() or col in ['u10', 'v10']]
                print(f"   天气特征数: {len(weather_cols)}")
            else:
                print(f"❌ 文件不存在: {file_path}")

if __name__ == "__main__":
    try:
        # 分析天气特征重要性
        analyze_weather_features_for_renewables()
        
        # 创建天气感知的新能源数据文件
        create_weather_aware_renewable_data()
        
        # 验证创建的文件
        validate_created_files()
        
        print("\n=== 天气感知新能源数据文件创建完成 ===")
        
    except Exception as e:
        print(f"创建天气感知新能源数据时出错: {e}")
        import traceback
        traceback.print_exc() 