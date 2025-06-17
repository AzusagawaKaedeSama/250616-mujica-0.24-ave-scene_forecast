#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
天气感知新能源预测模型训练脚本
支持光伏和风电的天气感知预测模型训练
基于train_torch.py中的train_weather_aware_model方法重构
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.torch_models import WeatherAwareConvTransformer
from data.dataset_builder import DatasetBuilder
from data.data_loader import DataLoader
from utils.scaler_manager import ScalerManager

def train_weather_aware_renewable_model(
    data_path: str,
    train_start_date: str = None,
    train_end_date: str = None,
    forecast_type: str = 'pv',
    weather_features: list = None,
    convtrans_config: dict = None,
    retrain: bool = False,
    dataset_id: str = None,
    peak_hours=(8, 20), 
    valley_hours=(0, 6), 
    peak_weight=2.5, 
    valley_weight=1.5,
    **kwargs
):
    """
    训练天气感知的新能源预测模型
    基于现有的高精度PeakAwareConvTransformer架构（仿照train_weather_aware_model）
    
    参数:
    data_path (str): 包含天气数据的CSV文件路径
    train_start_date (str): 训练开始日期
    train_end_date (str): 训练结束日期
    forecast_type (str): 预测类型，'pv' 或 'wind'
    weather_features (list): 天气特征列表
    convtrans_config (dict): 模型配置
    retrain (bool): 是否重新训练
    dataset_id (str): 数据集ID（省份）
    peak_hours, valley_hours: 峰谷时段定义（虽然对新能源不太适用，但保持一致性）
    peak_weight, valley_weight: 峰谷权重
    **kwargs: 其他参数
    
    返回:
    dict: 包含模型信息的字典
    """
    print(f"\n=== 训练天气感知模型 ({forecast_type} - {dataset_id}) ===")
    print(f"训练期间: {train_start_date} 到 {train_end_date}")
    
    # 验证预测类型
    if forecast_type not in ['pv', 'wind']:
        raise ValueError(f"不支持的预测类型: {forecast_type}，仅支持 'pv' 和 'wind'")
    
    # 根据预测类型确定数值列名
    value_column = forecast_type
    
    # 标准化路径（与负荷预测保持一致）
    model_base_name = 'convtrans_weather'
    model_dir = os.path.join('models', model_base_name, forecast_type, dataset_id)
    scaler_dir = os.path.join('models', 'scalers', model_base_name, forecast_type, dataset_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 初始化缩放器管理器
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 1. 加载包含天气数据的时间序列数据
    print(f"从 {data_path} 加载天气和{forecast_type}数据...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到天气数据文件: {data_path}")
    
    # 加载数据
    ts_data = pd.read_csv(data_path)
    
    # 处理时间列
    if 'datetime' in ts_data.columns:
        ts_data['datetime'] = pd.to_datetime(ts_data['datetime'])
        ts_data = ts_data.set_index('datetime')
    elif 'timestamp' in ts_data.columns:
        ts_data['timestamp'] = pd.to_datetime(ts_data['timestamp'])
        ts_data = ts_data.set_index('timestamp')
    else:
        raise ValueError("数据文件必须包含 'datetime' 或 'timestamp' 列")
    
    # 筛选训练日期范围
    train_data = ts_data.loc[train_start_date:train_end_date].copy()
    
    if train_data.empty:
        raise ValueError(f"在指定日期范围内没有找到数据: {train_start_date} - {train_end_date}")
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"数据列: {train_data.columns.tolist()}")
    
    # 2. 识别天气特征
    if weather_features is None:
        # 根据预测类型设置默认天气特征
        if forecast_type == 'pv':
            # 光伏预测重点关注：温度、湿度、降水、露点
            default_features = ['weather_temperature_c', 'weather_relative_humidity', 
                              'weather_precipitation_mm', 'weather_dewpoint_c']
        elif forecast_type == 'wind':
            # 风电预测重点关注：风速、温度、湿度
            default_features = ['weather_wind_speed', 'weather_temperature_c', 
                              'weather_relative_humidity']
        
        # 自动识别天气特征（排除已知的非天气列）
        non_weather_cols = [value_column, 'PARTY_ID', 'hour', 'day_of_week', 'month', 
                           'is_weekend', 'is_peak', 'is_valley']
        
        # 尝试使用默认特征，如果不存在则自动识别
        available_weather_features = [col for col in train_data.columns 
                                    if col not in non_weather_cols and 
                                    ('weather' in col.lower() or col.lower() in ['u10', 'v10'])]
        
        weather_features = []
        for feature in default_features:
            if feature in available_weather_features:
                weather_features.append(feature)
        
        # 如果默认特征不足，添加其他可用的天气特征
        if len(weather_features) < 3:  # 至少需要3个天气特征
            for feature in available_weather_features:
                if feature not in weather_features:
                    weather_features.append(feature)
                if len(weather_features) >= 6:  # 最多使用6个天气特征
                    break
    
    print(f"识别到的天气特征: {weather_features}")
    
    # 设置默认配置（与负荷预测保持一致）
    if convtrans_config is None:
        convtrans_config = {
            'seq_length': 96,
            'pred_length': 1,
            'batch_size': 32,
            'lr': 1e-4,
            'epochs': 50,
            'patience': 10,
            'use_peak_loss': True  # 保持与负荷预测一致，虽然对新能源意义不大
        }
    
    # 3. 使用现有的数据集构建器准备数据（包含峰值感知特征）
    data_loader = DataLoader()
    dataset_builder = DatasetBuilder(
        data_loader=data_loader,
        seq_length=convtrans_config['seq_length'],
        pred_horizon=1,
        standardize=False
    )
    
    print("准备训练数据（使用峰值感知特征工程 + 天气特征）...")
    
    # 使用现有的峰值感知数据准备方法（与负荷预测保持一致）
    X_train, y_train, X_val, y_val = dataset_builder.prepare_data_with_peak_awareness(
        ts_data=train_data,
        test_ratio=0.2, 
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=peak_weight,
        valley_weight=valley_weight,
        start_date=train_start_date,
        end_date=train_end_date,
        value_column=value_column
    )
    
    print(f"训练集 X 形状: {X_train.shape}, y 形状: {y_train.shape}")
    print(f"验证集 X 形状: {X_val.shape}, y 形状: {y_val.shape}")
    
    # 检查特征数量
    total_features = X_train.shape[2]
    base_features = 3  # 基础时间特征
    weather_feature_count = len(weather_features)
    peak_features = total_features - base_features - weather_feature_count
    
    print(f"特征分析: 总特征数={total_features}, 基础特征={base_features}, "
          f"天气特征={weather_feature_count}, 峰值特征={peak_features}")
    
    # 4. 标准化数据（使用现有的标准化逻辑）
    if not scaler_manager.has_scaler('X') or retrain:
        print("拟合 X 标准化器...")
        X_reshape = X_train.reshape(X_train.shape[0], -1)
        scaler_manager.fit('X', X_reshape)
    
    if not scaler_manager.has_scaler('y') or retrain:
        print("拟合 y 标准化器...")
        y_train_reshaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        scaler_manager.fit('y', y_train_reshaped)
    
    # 应用标准化
    print("应用标准化...")
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_scaled = scaler_manager.transform('X', X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    
    X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
    X_val_scaled = scaler_manager.transform('X', X_val_reshaped)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    y_train_shaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
    y_train_scaled = scaler_manager.transform('y', y_train_shaped)
    
    y_val_shaped = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
    y_val_scaled = scaler_manager.transform('y', y_val_shaped)
    
    # 确保 y 是一维的
    y_train_scaled_flat = y_train_scaled.flatten()
    y_val_scaled_flat = y_val_scaled.flatten()
    
    # 5. 创建天气感知模型
    input_shape = X_train_scaled.shape[1:]
    
    model_config_train = convtrans_config.copy()
    model_config_train['use_peak_loss'] = True  # 启用峰值感知损失（保持一致性）
    
    print("创建天气感知模型...")
    weather_model = WeatherAwareConvTransformer(
        input_shape=input_shape, 
        weather_features=weather_features,
        **model_config_train
    )
    
    # 6. 训练模型
    print(f"训练天气感知{forecast_type}模型...")
    
    try:
        weather_model.train_with_weather_awareness(
            X_train_scaled, y_train_scaled_flat,
            X_val_scaled, y_val_scaled_flat,
            epochs=convtrans_config['epochs'],
            batch_size=convtrans_config['batch_size'],
            save_dir=model_dir
        )
        
        # 保存模型
        weather_model.save(save_dir=model_dir)
        
        # 保存标准化器
        try:
            if scaler_manager.has_scaler('X'):
                scaler_manager.save_scaler('X', scaler_manager.get_scaler('X'))
            if scaler_manager.has_scaler('y'):
                scaler_manager.save_scaler('y', scaler_manager.get_scaler('y'))
            print(f"模型和缩放器已保存到目录: {model_dir} 和 {scaler_dir}")
        except Exception as save_err:
            print(f"保存标准化器时出错: {save_err}")
        
        print(f"天气感知{forecast_type}模型训练完成！")
        
    except Exception as e:
        print(f"模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    return {
        'model': weather_model,
        'weather_features': weather_features,
        'model_dir': model_dir,
        'scaler_dir': scaler_dir,
        'forecast_type': forecast_type
    }


def main():
    """主函数，用于测试训练功能"""
    
    # 测试参数
    test_province = '上海'
    test_forecast_type = 'wind'  # 测试风电模型
    
    # 构建数据路径
    data_path = f"data/timeseries_{test_forecast_type}_weather_{test_province}.csv"
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        print("请先运行 create_weather_aware_renewable_data.py 创建天气感知数据文件")
        return
    
    # 配置参数（与负荷预测保持一致）
    convtrans_config = {
        'seq_length': 96,
        'pred_length': 1,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 50,  # 测试时使用较少的epoch
        'patience': 10,
        'use_peak_loss': True
    }
    
    # 训练模型
    try:
        result = train_weather_aware_renewable_model(
            data_path=data_path,
            train_start_date='2024-01-01',
            train_end_date='2024-08-31',
            forecast_type=test_forecast_type,
            weather_features=None,  # 使用默认特征
            convtrans_config=convtrans_config,
            retrain=True,  # 强制重新训练
            dataset_id=test_province
        )
        
        print("\n训练结果:")
        print(f"模型目录: {result['model_dir']}")
        print(f"标准化器目录: {result['scaler_dir']}")
        print(f"天气特征: {result['weather_features']}")
        print(f"预测类型: {result['forecast_type']}")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 