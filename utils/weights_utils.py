import numpy as np
import pandas as pd

def get_time_based_weights(timestamps, peak_hours=(8, 20), peak_weight=2.0, weekend_weight=0.8):
    """
    基于时间生成预测权重
    
    参数:
    timestamps: 时间戳数组
    peak_hours: 高峰时段范围
    peak_weight: 工作日高峰时段权重
    weekend_weight: 周末权重
    
    返回:
    numpy.ndarray: 权重数组
    """
    weights = np.ones(len(timestamps))
    
    for i, ts in enumerate(timestamps):
        if isinstance(ts, np.datetime64):
            ts = pd.Timestamp(ts)
        
        # 工作日判断
        is_weekend = ts.dayofweek >= 5  # 5=周六, 6=周日
        
        # 高峰时段判断
        is_peak_hour = peak_hours[0] <= ts.hour <= peak_hours[1]
        
        # 设置权重
        if is_weekend:
            weights[i] = weekend_weight
        elif is_peak_hour:
            weights[i] = peak_weight
            
            # 根据具体时间微调权重
            # 例如，早晨高峰(8-10)和下午高峰(17-19)权重更高
            if 8 <= ts.hour <= 10 or 17 <= ts.hour <= 19:
                weights[i] *= 1.2  # 增加20%权重
    
    return weights

def dynamic_weight_adjustment(timestamp, peak_pred, non_peak_pred, typical_days_df, ts_data):
    """根据典型日模式动态调整高峰和非高峰模型的权重"""
    # 获取当前月份和日类型
    month = timestamp.month
    day_type = 'Weekend' if timestamp.dayofweek >= 5 else 'Workday'
    hour = timestamp.hour
    
    # 查找对应的典型日
    match = typical_days_df[(typical_days_df['group'] == month) & 
                          (typical_days_df['day_type'] == day_type)]
    
    if len(match) > 0:
        typical_day = match.iloc[0]['typical_day']
        
        # 找到典型日的负荷曲线
        typical_data = ts_data[ts_data.index.date == typical_day]
        
        # 计算当前时刻在典型日中的负荷位置（相对于一天中的最高和最低负荷）
        daily_min = typical_data['load'].min()
        daily_max = typical_data['load'].max()
        
        # 找到最接近当前时刻的典型日数据点
        closest_time = typical_data.index.get_indexer([timestamp], method='nearest')[0]
        if closest_time >= 0:
            load_level = typical_data.iloc[closest_time]['load']
            # 计算负荷水平的相对位置 (0-1之间)
            relative_level = (load_level - daily_min) / (daily_max - daily_min)
            
            # 根据负荷水平动态调整权重
            weight_peak = 0.3 + (0.6 * relative_level)  # 0.3-0.9的范围
            weight_non_peak = 1 - weight_peak
            
            return weight_peak * peak_pred + weight_non_peak * non_peak_pred, weight_peak, weight_non_peak
    
    # 默认权重策略
    if 8 <= hour <= 20:
        weight_peak = 0.7
        weight_non_peak = 0.3
    else:
        weight_peak = 0.3
        weight_non_peak = 0.7
    
    return weight_peak * peak_pred + weight_non_peak * non_peak_pred, weight_peak, weight_non_peak