#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气数据与负荷数据整合脚本
整合每个城市2024年的天气数据和该城市所在省份的负荷数据
在data文件夹下生成包含天气数据的各省份的timeseries csv文件
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 城市与省份映射关系
CITY_PROVINCE_MAPPING = {
    'shanghai': '上海',
    'nanjing': '江苏', 
    'hefei': '安徽',
    'hangzhou': '浙江',
    'fuzhou': '福建'
}

# 城市目录映射关系（用户提供的具体文件夹名称）
CITY_DIR_MAPPING = {
    'shanghai': 'shanghai_lon121.47_lat31.23',
    'nanjing': 'nanjing_lon118.78_lat32.06', 
    'hefei': 'hefei_lon117.27_lat31.86',
    'hangzhou': 'hangzhou_lon120.16_lat30.29',
    'fuzhou': 'fuzhou_lon119.30_lat26.08'
}

# 天气数据字段映射
WEATHER_FIELD_MAPPING = {
    'u10': 'u_wind_10m',      # 10米高度u风分量 (m/s)
    'v10': 'v_wind_10m',      # 10米高度v风分量 (m/s)
    'd2m': 'dewpoint_2m',     # 2米高度露点温度 (K)
    't2m': 'temperature_2m',  # 2米高度温度 (K)
    'tp': 'total_precipitation' # 总降水量 (m)
}

def kelvin_to_celsius(temp_k):
    """将开尔文温度转换为摄氏度"""
    return temp_k - 273.15

def calculate_wind_speed(u_wind, v_wind):
    """计算风速"""
    return np.sqrt(u_wind**2 + v_wind**2)

def calculate_relative_humidity(temp_k, dewpoint_k):
    """根据温度和露点温度计算相对湿度"""
    # 使用Magnus公式计算相对湿度
    temp_c = kelvin_to_celsius(temp_k)
    dewpoint_c = kelvin_to_celsius(dewpoint_k)
    
    # Magnus公式参数
    a = 17.27
    b = 237.7
    
    # 计算饱和水汽压
    es_temp = 6.112 * np.exp((a * temp_c) / (b + temp_c))
    es_dewpoint = 6.112 * np.exp((a * dewpoint_c) / (b + dewpoint_c))
    
    # 相对湿度 = 实际水汽压 / 饱和水汽压 * 100
    rh = (es_dewpoint / es_temp) * 100
    return np.clip(rh, 0, 100)  # 限制在0-100%之间

def calculate_hdd_cdd(temp_c, base_temp=18):
    """计算供暖度日(HDD)和制冷度日(CDD)"""
    hdd = np.maximum(base_temp - temp_c, 0)
    cdd = np.maximum(temp_c - base_temp, 0)
    return hdd, cdd

def calculate_temp_change_rate(temp_series):
    """计算温度变化率"""
    temp_change = temp_series.diff()
    return temp_change.fillna(0)

def load_weather_data_for_city(city_dir):
    """加载某个城市的全年天气数据"""
    logger.info(f"正在加载城市天气数据: {city_dir}")
    
    weather_data_list = []
    
    # 遍历所有月份目录
    for month in range(1, 13):
        month_dir = os.path.join(city_dir, f"2024-{month:02d}")
        if not os.path.exists(month_dir):
            logger.warning(f"月份目录不存在: {month_dir}")
            continue
            
        # 查找CSV文件
        csv_files = glob.glob(os.path.join(month_dir, "*.csv"))
        if not csv_files:
            logger.warning(f"在目录 {month_dir} 中未找到CSV文件")
            continue
            
        csv_file = csv_files[0]  # 取第一个CSV文件
        logger.info(f"加载文件: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            weather_data_list.append(df)
        except Exception as e:
            logger.error(f"加载文件失败 {csv_file}: {e}")
            continue
    
    if not weather_data_list:
        logger.error(f"未能加载任何天气数据: {city_dir}")
        return None
    
    # 合并所有月份数据
    weather_df = pd.concat(weather_data_list, ignore_index=True)
    
    # 转换时间格式
    weather_df['datetime'] = pd.to_datetime(weather_df['valid_time'])
    weather_df = weather_df.sort_values('datetime').reset_index(drop=True)
    
    logger.info(f"成功加载天气数据，共 {len(weather_df)} 条记录")
    return weather_df

def process_weather_data(weather_df):
    """处理天气数据，计算衍生指标"""
    logger.info("正在处理天气数据...")
    
    # 基础转换
    weather_df['weather_temperature_c'] = kelvin_to_celsius(weather_df['t2m'])
    weather_df['weather_dewpoint_c'] = kelvin_to_celsius(weather_df['d2m'])
    weather_df['weather_wind_speed'] = calculate_wind_speed(weather_df['u10'], weather_df['v10'])
    
    # 计算相对湿度
    weather_df['weather_relative_humidity'] = calculate_relative_humidity(
        weather_df['t2m'], weather_df['d2m']
    )
    
    # 计算HDD和CDD
    hdd, cdd = calculate_hdd_cdd(weather_df['weather_temperature_c'])
    weather_df['weather_HDD'] = hdd
    weather_df['weather_CDD'] = cdd
    
    # 计算温度变化率
    weather_df['weather_temp_change_rate'] = calculate_temp_change_rate(
        weather_df['weather_temperature_c']
    )
    
    # 降水量转换 (m -> mm)
    weather_df['weather_precipitation_mm'] = weather_df['tp'] * 1000
    
    # 选择需要的列
    processed_columns = [
        'datetime', 'weather_temperature_c', 'weather_wind_speed', 
        'weather_relative_humidity', 'weather_HDD', 'weather_CDD', 
        'weather_temp_change_rate', 'weather_precipitation_mm',
        'weather_dewpoint_c', 'u10', 'v10'
    ]
    
    return weather_df[processed_columns]

def interpolate_weather_to_15min(weather_df):
    """将小时级天气数据插值到15分钟间隔"""
    logger.info("正在将天气数据插值到15分钟间隔...")
    
    # 确保datetime列是datetime类型
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    
    # 去除重复的时间戳，保留第一个
    weather_df = weather_df.drop_duplicates(subset=['datetime'], keep='first')
    logger.info(f"去除重复时间戳后，剩余 {len(weather_df)} 条记录")
    
    # 设置datetime为索引
    weather_df = weather_df.set_index('datetime').sort_index()
    
    # 创建15分钟间隔的时间序列
    start_time = weather_df.index.min()
    end_time = weather_df.index.max()
    freq_15min = pd.date_range(start=start_time, end=end_time, freq='15min')
    
    # 对数值列进行线性插值
    numeric_cols = weather_df.select_dtypes(include=[np.number]).columns
    weather_15min = weather_df[numeric_cols].reindex(freq_15min).interpolate(method='linear')
    
    # 重置索引，将datetime作为列
    weather_15min = weather_15min.reset_index()
    weather_15min.rename(columns={'index': 'datetime'}, inplace=True)
    
    logger.info(f"插值完成，共 {len(weather_15min)} 条记录")
    return weather_15min

def load_load_data(province):
    """加载省份负荷数据"""
    load_file = f"../data/timeseries_load_{province}.csv"
    
    if not os.path.exists(load_file):
        logger.error(f"负荷数据文件不存在: {load_file}")
        return None
    
    logger.info(f"正在加载负荷数据: {load_file}")
    
    try:
        load_df = pd.read_csv(load_file)
        load_df['datetime'] = pd.to_datetime(load_df['datetime'])
        load_df = load_df.sort_values('datetime').reset_index(drop=True)
        logger.info(f"成功加载负荷数据，共 {len(load_df)} 条记录")
        return load_df
    except Exception as e:
        logger.error(f"加载负荷数据失败: {e}")
        return None

def merge_weather_load_data(weather_df, load_df, province):
    """合并天气数据和负荷数据"""
    logger.info(f"正在合并 {province} 的天气数据和负荷数据...")
    
    # 基于时间戳合并数据
    merged_df = pd.merge(load_df, weather_df, on='datetime', how='inner')
    
    logger.info(f"合并完成，共 {len(merged_df)} 条记录")
    
    # 检查数据完整性
    missing_load = merged_df['load'].isna().sum()
    missing_weather = merged_df['weather_temperature_c'].isna().sum()
    
    if missing_load > 0:
        logger.warning(f"负荷数据缺失 {missing_load} 条")
    if missing_weather > 0:
        logger.warning(f"天气数据缺失 {missing_weather} 条")
    
    return merged_df

def save_integrated_data(merged_df, province):
    """保存整合后的数据"""
    output_file = f"../data/timeseries_load_weather_{province}.csv"
    
    logger.info(f"正在保存整合数据到: {output_file}")
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存数据
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"成功保存整合数据，文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        # 打印数据统计信息
        print(f"\n=== {province} 数据统计 ===")
        print(f"时间范围: {merged_df['datetime'].min()} 到 {merged_df['datetime'].max()}")
        print(f"数据条数: {len(merged_df)}")
        print(f"负荷范围: {merged_df['load'].min():.1f} - {merged_df['load'].max():.1f}")
        print(f"温度范围: {merged_df['weather_temperature_c'].min():.1f}°C - {merged_df['weather_temperature_c'].max():.1f}°C")
        print(f"风速范围: {merged_df['weather_wind_speed'].min():.1f} - {merged_df['weather_wind_speed'].max():.1f} m/s")
        print(f"相对湿度范围: {merged_df['weather_relative_humidity'].min():.1f}% - {merged_df['weather_relative_humidity'].max():.1f}%")
        
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        return False
    
    return True

def process_city_data(city_name):
    """处理单个城市的数据"""
    logger.info(f"\n{'='*50}")
    logger.info(f"开始处理城市: {city_name}")
    logger.info(f"{'='*50}")
    
    # 获取省份名称
    province = CITY_PROVINCE_MAPPING.get(city_name)
    if not province:
        logger.error(f"未找到城市 {city_name} 对应的省份")
        return False
    
    # 获取城市目录名称
    city_dir = CITY_DIR_MAPPING.get(city_name)
    if not city_dir:
        logger.error(f"未找到城市 {city_name} 对应的目录名称")
        return False
    
    # 检查目录是否存在
    if not os.path.exists(city_dir):
        logger.error(f"城市目录不存在: {city_dir}")
        return False
    
    # 1. 加载天气数据
    weather_df = load_weather_data_for_city(city_dir)
    if weather_df is None:
        return False
    
    # 2. 处理天气数据
    weather_processed = process_weather_data(weather_df)
    
    # 3. 插值到15分钟间隔
    weather_15min = interpolate_weather_to_15min(weather_processed)
    
    # 4. 加载负荷数据
    load_df = load_load_data(province)
    if load_df is None:
        return False
    
    # 5. 合并数据
    merged_df = merge_weather_load_data(weather_15min, load_df, province)
    
    # 6. 保存整合数据
    success = save_integrated_data(merged_df, province)
    
    if success:
        logger.info(f"✅ {city_name} ({province}) 数据处理完成")
    else:
        logger.error(f"❌ {city_name} ({province}) 数据处理失败")
    
    return success

def main():
    """主函数"""
    logger.info("开始天气数据与负荷数据整合任务")
    logger.info(f"工作目录: {os.getcwd()}")
    
    # 统计结果
    success_count = 0
    total_count = len(CITY_PROVINCE_MAPPING)
    
    # 处理每个城市
    for city_name in CITY_PROVINCE_MAPPING.keys():
        logger.info("\n" + "="*50)
        logger.info(f"开始处理城市: {city_name}")
        logger.info("="*50)
        
        try:
            process_city_data(city_name)
            success_count += 1
        except Exception as e:
            logger.error(f"处理城市 {city_name} 时发生错误: {e}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("任务完成！")
    logger.info(f"成功处理: {success_count}/{total_count} 个城市")
    logger.info("="*60)
    
    if success_count == total_count:
        logger.info("🎉 所有城市数据处理成功！")
    else:
        logger.warning(f"⚠️  有 {total_count - success_count} 个城市处理失败")

if __name__ == "__main__":
    main() 