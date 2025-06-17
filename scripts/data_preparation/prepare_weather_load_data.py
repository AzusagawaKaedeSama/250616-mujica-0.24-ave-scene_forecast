#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
准备包含气象数据的负荷预测数据集

此脚本加载福州的气象数据和福建省的负荷数据，将它们合并，
并生成一个增强型数据集用于负荷预测。
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入气象数据处理器
from utils.weather_processor import WeatherProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='准备包含气象数据的负荷预测数据集')
    
    parser.add_argument('--load_data', type=str, default=None,
                        help='负荷数据文件路径，如不指定将自动搜索')
    parser.add_argument('--weather_data_dir', type=str, default='data_preprocess',
                        help='气象数据目录，默认为 data_preprocess')
    parser.add_argument('--location', type=str, default='fuzhou_lon119.30_lat26.08',
                        help='气象数据位置标识符，默认为福州')
    parser.add_argument('--start_date', type=str, default='2024-01-01',
                        help='数据开始日期 (YYYY-MM-DD)，默认为 2024-01-01')
    parser.add_argument('--end_date', type=str, default='2024-12-31',
                        help='数据结束日期 (YYYY-MM-DD)，默认为 2024-01-31')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径，如不指定将自动生成')
    parser.add_argument('--dataset_id', type=str, default='fujian',
                        help='数据集标识符，用于查找对应的负荷数据文件，可以是拼音(fujian)或中文(福建)，默认为 fujian')
    
    return parser.parse_args()

def find_load_data_file(dataset_id='fujian'):
    """
    查找匹配指定数据集ID的负荷数据文件
    """
    # 检查数据目录
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return None
    
    # 首先尝试直接查找中文名称的文件
    if dataset_id == 'fujian':
        chinese_filename = 'timeseries_load_福建.csv'
        if os.path.exists(os.path.join(data_dir, chinese_filename)):
            return os.path.join(data_dir, chinese_filename)
    
    # 搜索负荷数据文件模式（拼音格式）
    possible_files = [
        f for f in os.listdir(data_dir) 
        if f.startswith(f'timeseries_load_{dataset_id}') and f.endswith('.csv')
    ]
    
    if possible_files:
        # 返回最新的文件(按字母顺序，假设文件名包含日期时间)
        return os.path.join(data_dir, sorted(possible_files)[-1])
    
    # 尝试查找中文名称的文件（如果dataset_id是拼音形式）
    possible_chinese_mappings = {
        'fujian': '福建',
        'shanghai': '上海',
        'beijing': '北京',
        # 可以根据需要添加更多映射
    }
    
    chinese_name = possible_chinese_mappings.get(dataset_id)
    if chinese_name:
        chinese_filename = f'timeseries_load_{chinese_name}.csv'
        if os.path.exists(os.path.join(data_dir, chinese_filename)):
            return os.path.join(data_dir, chinese_filename)
    
    # 如果没有找到特定数据集ID的文件，尝试查找任何负荷数据文件
    any_load_files = [
        f for f in os.listdir(data_dir) 
        if f.startswith('timeseries_load_') and f.endswith('.csv')
    ]
    
    if any_load_files:
        print(f"警告: 未找到{dataset_id}的负荷数据，使用{any_load_files[0]}")
        return os.path.join(data_dir, any_load_files[0])
    
    return None

def main():
    args = parse_args()
    
    # 查找负荷数据文件
    load_data_file = args.load_data
    if load_data_file is None:
        print(f"正在查找数据集ID为 '{args.dataset_id}' 的负荷数据文件...")
        load_data_file = find_load_data_file(args.dataset_id)
        if load_data_file is None:
            print(f"错误: 未找到负荷数据文件，请使用--load_data参数指定文件路径")
            return 1
    
    # 加载负荷数据
    print(f"加载负荷数据: {load_data_file}")
    try:
        load_data = pd.read_csv(load_data_file, index_col=0)
        load_data.index = pd.to_datetime(load_data.index)
        print(f"成功加载负荷数据，共 {len(load_data)} 行，时间范围: {load_data.index.min()} 到 {load_data.index.max()}")
    except Exception as e:
        print(f"加载负荷数据时出错: {e}")
        return 1
    
    # 确定日期范围
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    # 过滤负荷数据范围
    load_data = load_data.loc[start_date:end_date]
    print(f"过滤后的负荷数据范围: {start_date} 到 {end_date}，共 {len(load_data)} 行")
    
    # 确保有'load'列
    if 'load' not in load_data.columns:
        # 尝试查找包含'load'的列
        load_cols = [col for col in load_data.columns if 'load' in col.lower()]
        if load_cols:
            print(f"将列 '{load_cols[0]}' 重命名为 'load'")
            load_data = load_data.rename(columns={load_cols[0]: 'load'})
        else:
            # 使用第一列作为负荷
            first_col = load_data.columns[0]
            print(f"未找到负荷列，将第一列 '{first_col}' 作为负荷列")
            load_data = load_data.rename(columns={first_col: 'load'})
    
    # 初始化气象数据处理器
    print(f"初始化气象数据处理器，目录: {args.weather_data_dir}")
    weather_processor = WeatherProcessor(weather_data_dir=args.weather_data_dir)
    
    # 加载指定日期范围的气象数据
    print(f"加载气象数据，位置: {args.location}, 日期范围: {args.start_date} 到 {args.end_date}")
    
    # 确定需要加载的月份范围
    start_ym = start_date.strftime('%Y-%m')
    end_ym = end_date.strftime('%Y-%m')
    
    try:
        weather_data = weather_processor.load_multiple_months(
            location=args.location,
            start_year_month=start_ym,
            end_year_month=end_ym
        )
        print(f"成功加载气象数据，共 {len(weather_data)} 条记录")
    except Exception as e:
        print(f"加载气象数据时出错: {e}")
        return 1
    
    # 合并负荷和气象数据
    print("合并负荷和气象数据...")
    try:
        merged_data = weather_processor.merge_with_load_data(load_data)
        print(f"合并后的数据集共 {len(merged_data)} 条记录，特征列: {', '.join(merged_data.columns)}")
    except Exception as e:
        print(f"合并数据时出错: {e}")
        return 1
    
    # 保存合并后的数据
    output_file = args.output_file
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data/timeseries_load_weather_{args.dataset_id}_{timestamp}.csv"
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        merged_data.to_csv(output_file)
        print(f"已成功保存合并后的数据到: {output_file}")
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 