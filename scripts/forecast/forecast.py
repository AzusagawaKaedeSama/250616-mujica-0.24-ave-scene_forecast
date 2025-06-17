import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
import pickle
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')
import torch

# 设置标准输出编码为UTF-8，解决中文显示乱码问题
sys.stdout.reconfigure(encoding='utf-8')

from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder
from utils.evaluator import ModelEvaluator, plot_peak_forecast_analysis, calculate_metrics, plot_forecast_results
from utils.scaler_manager import ScalerManager
from models.torch_models import TorchConvTransformer, PeakAwareConvTransformer, WeatherAwareConvTransformer
import shutil
from utils.weights_utils import get_time_based_weights, dynamic_weight_adjustment
import json

def perform_combined_forecast(start_date, end_date, forecast_interval=15, 
                              peak_hours=(8, 20), peak_weight=2.0, apply_smoothing=False,dataset_id='上海'):
    """
    使用高峰模型和非高峰模型的动态加权平均进行预测
    """
    print(f"\n=== 使用动态加权平均的组合模型进行预测 ===", flush=True)
    print(f"预测期间: {start_date} 到 {end_date}", flush=True)
    print(f"预测间隔: {forecast_interval}分钟", flush=True)
    print(f"高峰时段: {peak_hours[0]}:00 - {peak_hours[1]}:00", flush=True)
    
    # 调用高峰模型预测
    print("调用高峰模型进行预测...", flush=True)
    peak_results = perform_rolling_forecast_with_peak_awareness(
        start_date=start_date,
        end_date=end_date,
        forecast_interval=forecast_interval,
        peak_hours=peak_hours,
        peak_weight=peak_weight,
        apply_smoothing=False,
        dataset_id=dataset_id
    )
    
    # 调用非高峰模型预测
    print("调用非高峰模型进行预测...", flush=True)
    non_peak_results = perform_rolling_forecast(
        start_date=start_date,
        end_date=end_date,
        forecast_interval=forecast_interval,
        apply_smoothing=False,
        dataset_id=dataset_id
    )
    
    # 合并结果
    print("合并高峰模型和非高峰模型的预测结果...", flush=True)
    combined_results = []
    for i in range(len(peak_results)):
        peak_pred = peak_results.iloc[i]['predicted']
        non_peak_pred = non_peak_results.iloc[i]['predicted']
        forecast_time = peak_results.iloc[i]['datetime']
        
        # 动态加权
        hour = pd.Timestamp(forecast_time).hour
        if peak_hours[0] <= hour <= peak_hours[1]:
            weight_peak = 1  # 高峰时段高峰模型权重
            weight_non_peak = 0
        else:
            if hour<peak_hours[0]-1 or hour>peak_hours[1]+1:
                # 非高峰时段非高峰模型权重
                weight_peak = 0
                weight_non_peak = 1
            else:
                # 过渡时段权重
                weight_peak = 0.3  
                weight_non_peak = 0.7
        
        combined_pred = (weight_peak * peak_pred + weight_non_peak * non_peak_pred)
        
        # 获取实际值
        actual_value = peak_results.iloc[i]['actual']
        
        # 添加到结果中
        combined_results.append({
            'datetime': forecast_time,
            'predicted': combined_pred,
            'actual': actual_value,
            'peak_pred': peak_pred,
            'non_peak_pred': non_peak_pred,
            'weight_peak': weight_peak,
            'weight_non_peak': weight_non_peak
        })
    
    # 创建结果数据框
    combined_results_df = pd.DataFrame(combined_results)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = "results/combined"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = f"{results_dir}/combined_forecast_{start_date}_{end_date}_{timestamp}.csv"
    combined_results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}", flush=True)
    
    return combined_results_df


def perform_rolling_forecast_with_peak_awareness(data_path, start_date, end_date, forecast_interval=15, 
                                             peak_hours=(8, 20), valley_hours=(0, 6),
                                             peak_weight=5.0, valley_weight=1.5,
                                             apply_smoothing=False, model_timestamp=None, dataset_id='上海', forecast_type='load',
                                             historical_days=8):
    """
    执行带有高峰感知的滚动预测
    
    Args:
        data_path: 时间序列数据路径
        start_date: 预测开始日期
        end_date: 预测结束日期
        forecast_interval: 预测时间间隔（分钟）
        peak_hours: 高峰时段起止小时
        valley_hours: 低谷时段起止小时
        peak_weight: 高峰权重
        valley_weight: 低谷权重
        apply_smoothing: 是否应用平滑
        model_timestamp: 模型时间戳
        dataset_id: 数据集ID
        forecast_type: 预测类型
        historical_days: 用于模型输入的历史数据天数，默认为8天
    
    Returns:
        预测结果DataFrame
    """
    
    # 根据预测类型选择合适的目录
    value_column = forecast_type # Column name based on type
    
    model_dir = f"models/convtrans_peak/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/convtrans_peak/{forecast_type}/{dataset_id}"
    results_dir = f"results/convtrans_peak/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"使用数据: {data_path}", flush=True)
    print(f"预测类型: {forecast_type}, 数据集: {dataset_id}", flush=True)
    print(f"模型目录: {model_dir}", flush=True)
    print(f"使用 {historical_days} 天的历史数据用于特征提取", flush=True)
    
    # 检查模型是否存在
    if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir)):
        raise FileNotFoundError(f"模型目录不存在或其中没有模型文件: {model_dir}")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"缩放器目录不存在: {scaler_dir}")
    
    # 加载模型和缩放器
    print(f"从 {model_dir} 加载模型...", flush=True)
    model = PeakAwareConvTransformer.load(save_dir=model_dir)
    
    # 从模型配置获取seq_length
    seq_length = model.config.get('seq_length', 96)
    
    print(f"从 {scaler_dir} 加载缩放器...", flush=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 加载时间序列数据
    ts_data_path = data_path
    if not os.path.exists(ts_data_path):
        raise FileNotFoundError(f"时间序列数据文件不存在: {ts_data_path}")

    print(f"从 {ts_data_path} 加载时间序列数据...", flush=True)
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据集构建器
    dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1)
    
    # 设置预测时间范围
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # 扩展历史数据范围
    # 需要 seq_length 点来做第一次预测，并且 build_dataset_with_peak_awareness 可能需要更早的数据来计算特征
    # 保守起见，向前扩展 historical_days 天
    required_hist_start = start_datetime - timedelta(minutes=seq_length * forecast_interval)
    extended_history_start = required_hist_start - timedelta(days=historical_days) 
    if ts_data.index.min() > extended_history_start:
        print(f"警告: 历史数据可能不足以计算所有特征，最早数据点: {ts_data.index.min()}，需要: {extended_history_start}", flush=True)
        # 如果连基本的 seq_length 都不够，则报错
        if ts_data.index.min() > required_hist_start:
             raise ValueError(f"没有足够的历史数据。需要从 {required_hist_start} 开始的数据 (基于 seq_length={seq_length})")
        extended_history_start = ts_data.index.min() # 使用已有的最早数据
    
    # 获取扩展后的数据范围
    extended_data = ts_data.loc[extended_history_start:end_datetime].copy()
    
    # 构建预测时间点列表
    forecast_times = []
    current_time = start_datetime
    while current_time <= end_datetime:
        forecast_times.append(current_time)
        current_time += timedelta(minutes=forecast_interval)
    
    # 准备结果容器
    results = []

    # 对每个时间点进行预测
    print(f"开始滚动预测，共 {len(forecast_times)} 个时间点...", flush=True)
    for i, forecast_time in enumerate(forecast_times):
        if i % 50 == 0:  # 每50次预测显示一次进度
            print(f"正在预测 {i+1}/{len(forecast_times)}: {forecast_time}", flush=True)
        
        try:
            # 准备输入数据 - 使用 build_dataset_with_peak_awareness
            current_hist_end = forecast_time - timedelta(minutes=forecast_interval)
            
            # 动态确定用于特征工程的数据范围
            feature_eng_start = max(extended_history_start, current_hist_end - timedelta(days=8)) # 例如，最多用前8天数据计算特征

            enhanced_data = dataset_builder.build_dataset_with_peak_awareness(
                df=extended_data.loc[feature_eng_start:current_hist_end], # 限制用于特征工程的数据量
                date_column=None, # 已经是索引
                value_column=value_column, 
                interval=forecast_interval,
                peak_hours=peak_hours,
                valley_hours=valley_hours,
                peak_weight=peak_weight,
                valley_weight=valley_weight,
                start_date=feature_eng_start, # 传递正确的开始日期
                end_date=current_hist_end # 传递正确的结束日期
            )
            
            # 确保有足够的数据点用于最后的序列提取
            if len(enhanced_data) < seq_length:
                 raise ValueError(f"为 {forecast_time} 准备数据时，增强后的数据点数不足 ({len(enhanced_data)} < {seq_length})，请检查数据范围和特征工程")
            
            # 提取最近的历史数据 (seq_length 个点)
            X = enhanced_data.iloc[-seq_length:].drop(columns=[value_column]).values
            
            # 确保特征维度正确
            expected_features = model.config.get('input_shape')[-1] if hasattr(model, 'config') and 'input_shape' in model.config else X.shape[1]
            if X.shape[1] != expected_features:
                 print(f"警告: 特征维度不匹配！预期 {expected_features}, 得到 {X.shape[1]}", flush=True)
                 # 可以尝试填充或截断，但这通常表明数据准备有问题
                 # X = X[:, :expected_features] # 简单截断示例
            
            X = X.reshape(1, seq_length, -1)  # 调整为 [1, seq, features]
            
            # 标准化输入数据
            X_scaled = scaler_manager.transform('X', X.reshape(1, -1)).reshape(X.shape)
            
            # 预测
            raw_pred = model.predict(X_scaled)
            
            pred_inverse = scaler_manager.inverse_transform('y', raw_pred)
            predicted_value = pred_inverse.flatten()[0]
            
            # 获取实际值
            actual_value = np.nan
            if forecast_time in ts_data.index:
                actual_value = ts_data.loc[forecast_time, value_column]
            
            # 添加到结果中
            results.append({
                'datetime': forecast_time,
                'predicted': predicted_value,
                'actual': actual_value,
                'is_peak': forecast_time.hour >= peak_hours[0] and forecast_time.hour <= peak_hours[1]
            })
        
        except Exception as e:
            print(f"预测 {forecast_time} 时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results.append({
                'datetime': forecast_time,
                'predicted': np.nan,
                'actual': ts_data.loc[forecast_time, value_column] if forecast_time in ts_data.index else np.nan,
                'is_peak': forecast_time.hour >= peak_hours[0] and forecast_time.hour <= peak_hours[1]
            })
    
    # 创建结果数据框
    results_df = pd.DataFrame(results, columns=['datetime', 'actual', 'predicted', 'is_peak'])
    
    # 应用平滑处理（如果启用）
    if apply_smoothing and len(results_df) > 0 and not results_df['predicted'].isna().all():
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].fillna(method='ffill').fillna(method='bfill').values,
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    else:
        results_df['predicted_smoothed'] = results_df['predicted']
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/peak_aware_forecast_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}", flush=True)
    
    return results_df

def perform_rolling_forecast_with_non_peak_awareness(data_path, start_date, end_date, forecast_interval=15, 
                                             peak_hours=(8, 20), valley_hours=(0, 6),
                                             peak_weight=5.0, valley_weight=1.5,
                                             apply_smoothing=False, model_timestamp=None,dataset_id='上海', forecast_type='load'):
    """
    使用非高峰感知模型（但可能是用高峰感知特征工程训练的）进行日内滚动预测
    """
    value_column = forecast_type
    print(f"\n=== 使用非高峰感知模型进行日内{forecast_type}滚动预测 ({dataset_id}) ===", flush=True)
    print(f"预测期间: {start_date} 到 {end_date}", flush=True)
    print(f"预测间隔: {forecast_interval}分钟", flush=True)
    print(f"高峰时段: {peak_hours[0]}:00 - {peak_hours[1]}:00", flush=True)
    print(f"平滑处理: {'启用' if apply_smoothing else '禁用'}", flush=True)
    print(f"预测类型: {forecast_type}", flush=True)
    
    # --- 更新: 使用包含 forecast_type 的路径 --- 
    model_base_name = 'convtrans' # 基础模型名称 (假设非高峰模型保存在这里)
    model_dir_base = f"models/{model_base_name}/{forecast_type}/{dataset_id}"
    results_dir = f"results/{model_base_name}/{forecast_type}/{dataset_id}" # 结果可以放基础模型下
    scaler_dir_base = f"models/scalers/{model_base_name}/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)
    # ------------------------------------------

    # 如果指定了时间戳... (处理 model_timestamp)
    if model_timestamp:
        model_dir = f"{model_dir_base}/{model_timestamp}"
        scaler_dir = f"{scaler_dir_base}/{model_timestamp}"
    else:
        model_dir = model_dir_base
        scaler_dir = scaler_dir_base
    
    # 检查模型和缩放器是否存在
    if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir)):
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在或其中没有 .pth 模型文件")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"缩放器目录不存在: {scaler_dir}")
    
    print(f"从 {model_dir} 加载已训练的模型...", flush=True)
    model = PeakAwareConvTransformer.load(save_dir=model_dir) # 加载模型
    seq_length = model.config.get('seq_length', int(1440/forecast_interval))

    print(f"从 {scaler_dir} 加载缩放器...", flush=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 加载时间序列数据
    ts_data_path = data_path
    if not os.path.exists(ts_data_path):
        raise FileNotFoundError(f"时间序列数据文件不存在: {ts_data_path}")

    print(f"从 {ts_data_path} 加载时间序列数据...", flush=True)
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据集构建器
    dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1, standardize=False)
    
    # 确保有足够的历史数据
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # 扩展历史数据范围（向前扩展7天）
    extended_history_start = start_datetime - timedelta(days=7)
    if ts_data.index.min() > extended_history_start:
        raise ValueError(f"没有足够的历史数据。需要从 {extended_history_start} 开始的数据")
    
    # 获取扩展后的数据范围
    extended_data = ts_data.loc[extended_history_start:end_datetime].copy()
    
    # 构建预测时间点列表
    forecast_times = []
    current_time = start_datetime
    while current_time <= end_datetime:
        forecast_times.append(current_time)
        current_time += timedelta(minutes=forecast_interval)
    
    # 准备结果容器
    results = []
    
    # 对每个时间点进行预测
    print(f"开始滚动预测，共 {len(forecast_times)} 个时间点...", flush=True)
    for i, forecast_time in enumerate(forecast_times):
        if i % 50 == 0:  # 每50次预测显示一次进度
            print(f"正在预测 {i+1}/{len(forecast_times)}: {forecast_time}", flush=True)
        
        try:
            # 准备输入数据
            enhanced_data = dataset_builder.build_dataset_with_peak_awareness(
                df=extended_data,
                date_column='datetime',
                value_column=value_column,
                peak_hours=peak_hours,
                valley_hours=valley_hours,
                peak_weight=peak_weight,
                valley_weight=valley_weight,
                start_date=extended_history_start,
                end_date=forecast_time
            )
            # print(f"增强数据集生成成功，形状: {enhanced_data.shape}")
            
            # 提取最近的历史数据
            X = enhanced_data.iloc[-dataset_builder.seq_length:].drop(columns=[value_column]).values
            # print(f"输入数据形状: {X.shape}")
            
            X = X.reshape(1, X.shape[0], X.shape[1])  # 添加批次维度
            # print(f"重塑后的输入数据形状: {X.shape}")
            
            # 标准化输入数据
            X_scaled = scaler_manager.transform('X', X.reshape(1, -1)).reshape(X.shape)
            # print(f"标准化后的输入数据形状: {X_scaled.shape}")
            
            # 预测
            raw_pred = model.predict(X_scaled)
            # print(f"模型预测成功，原始预测值: {raw_pred}")
            
            pred_inverse = scaler_manager.inverse_transform('y', raw_pred)
            predicted_value = pred_inverse.flatten()[0]
            # print(f"反标准化后的预测值: {predicted_value}")
            
            # 获取实际值
            actual_value = np.nan
            if forecast_time in ts_data.index:
                actual_value = ts_data.loc[forecast_time, value_column]
            
            # 添加到结果中
            results.append({
                'datetime': forecast_time,
                'predicted': predicted_value,
                'actual': actual_value,
                'is_peak': enhanced_data.iloc[-1]['is_peak']
            })
        
        except Exception as e:
            print(f"预测 {forecast_time} 时出错: {e}", flush=True)
            # 继续下一个时间点的预测
    
    # 创建结果数据框
    results_df = pd.DataFrame(results, columns=['datetime', 'actual', 'predicted', 'is_peak'])
    
    # 应用平滑处理（如果启用）
    if apply_smoothing and len(results_df) > 0:
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].values,
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    else:
        results_df['predicted_smoothed'] = results_df['predicted']
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/peak_aware_forecast_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}", flush=True)
    
    return results_df

# 调试：打印模型期望的输入形状
def inspect_model_input_shape(model):
    """检查模型期望的输入形状"""
    print("\n=== 模型输入形状检查 ===", flush=True)
    if hasattr(model.forecaster, 'model'):
        model_impl = model.forecaster.model
        
        # 查看模型第一层的输入形状
        if hasattr(model_impl, 'spatial_conv'):
            conv_layer = model_impl.spatial_conv[0]
            print(f"第一个卷积层: {conv_layer}", flush=True)
            if hasattr(conv_layer, 'in_channels'):
                print(f"输入通道数: {conv_layer.in_channels}", flush=True)
            if hasattr(conv_layer, 'weight'):
                print(f"权重形状: {conv_layer.weight.shape}", flush=True)
        
        # 尝试执行前向传播，获取输入形状要求
        try:
            # 创建一个示例输入，测试不同的形状
            test_shape1 = (1, 37, 96)  # [batch, features, seq]
            test_shape2 = (1, 96, 37)  # [batch, seq, features]
            
            # 尝试形状1
            try:
                test_input1 = torch.zeros(test_shape1, device=model.forecaster.device)
                output1 = model_impl(test_input1)
                print(f"形状 {test_shape1} 工作正常，输出形状: {output1.shape}", flush=True)
            except Exception as e:
                print(f"形状 {test_shape1} 不兼容: {e}", flush=True)
            
            # 尝试形状2
            try:
                test_input2 = torch.zeros(test_shape2, device=model.forecaster.device)
                output2 = model_impl(test_input2)
                print(f"形状 {test_shape2} 工作正常，输出形状: {output2.shape}", flush=True)
            except Exception as e:
                print(f"形状 {test_shape2} 不兼容: {e}", flush=True)
            
        except Exception as e:
            print(f"测试输入形状时出错: {e}", flush=True)
    else:
        print("无法访问模型实现", flush=True)

def perform_rolling_forecast(data_path, start_date, end_date, forecast_interval=5, apply_smoothing=False, dataset_id='上海', forecast_type='load', historical_days=8):
    """
    执行标准滚动预测
    
    Args:
        data_path: 时间序列数据路径
        start_date: 预测开始日期
        end_date: 预测结束日期  
        forecast_interval: 预测时间间隔（分钟）
        apply_smoothing: 是否应用平滑
        dataset_id: 数据集ID
        forecast_type: 预测类型
        historical_days: 用于模型输入的历史数据天数，默认为8天
        
    Returns:
        预测结果DataFrame
    """
    
    # 根据预测类型选择合适的目录
    value_column = forecast_type
    
    model_dir = f"models/convtrans/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/convtrans/{forecast_type}/{dataset_id}"
    results_dir = f"results/convtrans/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"使用数据: {data_path}", flush=True)
    print(f"预测类型: {forecast_type}, 数据集: {dataset_id}", flush=True)
    print(f"使用 {historical_days} 天的历史数据用于特征提取", flush=True)
    
    # 检查模型是否存在
    if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir)):
        raise FileNotFoundError(f"模型目录不存在或其中没有模型文件: {model_dir}")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"缩放器目录不存在: {scaler_dir}")
    
    # 加载模型和缩放器
    model = TorchConvTransformer.load(save_dir=model_dir)
    
    # 从模型配置获取seq_length
    seq_length = model.config.get('seq_length', 96)
    print(f"使用seq_length={seq_length}", flush=True)
    
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 加载时间序列数据
    ts_data_path = data_path
    if not os.path.exists(ts_data_path):
        raise FileNotFoundError(f"时间序列数据文件不存在: {ts_data_path}")

    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据集构建器
    dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1)
    
    # 设置预测时间范围
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # 准备用于评估的实际数据
    actual_data = ts_data.copy()
    
    # 检查是否有足够的历史数据
    # 扩展历史数据范围，使用historical_days参数
    required_hist_start = start_datetime - timedelta(minutes=seq_length * forecast_interval)
    extended_history_start = required_hist_start - timedelta(days=historical_days)
    
    if ts_data.index.min() > extended_history_start:
        print(f"警告: 历史数据可能不足以计算所有特征，最早数据点: {ts_data.index.min()}，需要: {extended_history_start}", flush=True)
        # 如果连基本的 seq_length 都不够，则报错
        if ts_data.index.min() > required_hist_start:
            raise ValueError(f"没有足够的历史数据。需要从 {required_hist_start} 开始的数据 (基于 seq_length={seq_length})")
        extended_history_start = ts_data.index.min() # 使用已有的最早数据
    
    print(f"使用从 {extended_history_start} 到 {end_datetime} 的数据范围", flush=True)
    
    # 构建预测时间点列表
    forecast_times = []
    current_time = start_datetime
    while current_time <= end_datetime:
        forecast_times.append(current_time)
        current_time += timedelta(minutes=forecast_interval)
    
    # 准备结果容器
    results = []
    
    # 保存最近的实际负荷数据，用于波动性计算 (如果需要)
    # recent_loads = [] ... (这部分逻辑可以保留，但可能不是所有预测类型都需要)
    
    # 对每个时间点进行预测
    print(f"开始滚动预测，共 {len(forecast_times)} 个时间点...", flush=True)
    for i, forecast_time in enumerate(forecast_times):
        if i % 50 == 0:  # 每50次预测显示一次进度
            print(f"正在预测 {i+1}/{len(forecast_times)}: {forecast_time}", flush=True)
            
        try:
            # --- 更新: 使用 build_dataset_with_peak_awareness 准备输入 --- 
            # 以确保特征与训练时一致
            current_hist_end = forecast_time - timedelta(minutes=forecast_interval)
            current_hist_start = current_hist_end - timedelta(minutes=(seq_length-1)*forecast_interval)
            
            # 注意：这里的 peak/valley hours/weights 需要传递，或者从模型配置获取
            # 暂时使用默认值，理想情况应该与训练时一致
            temp_peak_hours = (8, 20) 
            temp_valley_hours = (0, 6)
            temp_peak_weight = 2.0
            temp_valley_weight = 1.5
            
            enhanced_data = dataset_builder.build_dataset_with_peak_awareness(
                df=ts_data[ts_data.index <= current_hist_end], # 只取到预测点之前的数据
                date_column=None, # 已经是索引
                value_column=value_column,
                interval=forecast_interval,
                peak_hours=temp_peak_hours,
                valley_hours=temp_valley_hours,
                peak_weight=temp_peak_weight,
                valley_weight=temp_valley_weight,
                start_date=current_hist_start, # 确保范围正确
                end_date=current_hist_end
            )
            
            if len(enhanced_data) < seq_length:
                 raise ValueError(f"为 {forecast_time} 准备数据时，点数不足 ({len(enhanced_data)} < {seq_length})")

            X = enhanced_data.iloc[-seq_length:].drop(columns=[value_column]).values
            X = X.reshape(1, seq_length, -1) # 调整为 [1, seq, features]
            # -----------------------------------------------------------------
            
            # 标准化输入数据
            X_reshaped = X.reshape(X.shape[0], -1)
            X_scaled = scaler_manager.transform('X', X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # 进行预测
            raw_pred = model.predict(X_scaled)
            
            # 确保形状正确
            raw_pred_shaped = raw_pred.reshape(-1, 1) if len(raw_pred.shape) == 1 else raw_pred
            
            # 反向标准化
            pred_inverse = scaler_manager.inverse_transform('y', raw_pred_shaped)
            predicted_value = pred_inverse.flatten()[0]
            
            # 获取实际值（如果有）
            actual_value = np.nan
            if forecast_time in actual_data.index:
                actual_value = actual_data.loc[forecast_time, value_column]
                # 更新最近负荷历史... (如果需要)
            
            # 添加到结果中
            results.append({
                'datetime': forecast_time,
                'predicted': predicted_value,
                'actual': actual_value
            })
            
        except Exception as e:
            print(f"预测 {forecast_time} 时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # 如果预测失败，可以添加一个 NaN 或其他标记
            results.append({
                'datetime': forecast_time,
                'predicted': np.nan,
                'actual': actual_data.loc[forecast_time, value_column] if forecast_time in actual_data.index else np.nan
            })
    
    # 创建结果数据框
    results_df = pd.DataFrame(results)
    
    # 应用自适应平滑处理（如果启用）
    if apply_smoothing and len(results_df) > 0 and not results_df['predicted'].isna().all():
        # 平滑预测结果
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].fillna(method='ffill').fillna(method='bfill').values, # 填充 NaN 再平滑
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    else:
        # 如果不应用平滑，将原始预测值复制到平滑列
        results_df['predicted_smoothed'] = results_df['predicted']
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/rolling_forecast_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}", flush=True)
    
    # 计算指标（仅对有实际值的时间点）
    valid_results = results_df.dropna(subset=['actual', 'predicted'])
    
    if len(valid_results) > 0:
        # 计算原始预测的指标
        metrics_orig = calculate_metrics(valid_results['actual'], valid_results['predicted'])
        
        # 计算平滑后预测的指标
        metrics_smooth = calculate_metrics(valid_results['actual'], valid_results['predicted_smoothed'])
        
        print(f"原始预测指标:", flush=True)
        for k, v in metrics_orig.items(): print(f"  {k}: {v:.4f}", flush=True)
        
        if apply_smoothing:
            print(f"\n平滑后预测指标:", flush=True)
            for k, v in metrics_smooth.items(): print(f"  {k}: {v:.4f}", flush=True)
        
        # 可以考虑保存指标到文件
        
    else:
        print("没有足够的有效数据来计算指标。", flush=True)

    return results_df

def perform_rolling_forecast_with_patterns(data_path, start_date, end_date, forecast_interval=15, dataset_id='上海', forecast_type='load', historical_days=8):
    """
    执行基于模式识别的滚动预测
    
    Args:
        data_path: 时间序列数据路径
        start_date: 预测开始日期
        end_date: 预测结束日期
        forecast_interval: 预测时间间隔（分钟）
        dataset_id: 数据集ID
        forecast_type: 预测类型
        historical_days: 用于模型输入的历史数据天数，默认为8天
        
    Returns:
        预测结果DataFrame
    """
    
    # 根据预测类型选择合适的目录
    value_column = forecast_type 
    
    model_dir = f"models/pattern_specific/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/pattern_specific/{forecast_type}/{dataset_id}"
    results_dir = f"results/pattern_forecast/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    scaler_manager = ScalerManager(scaler_path=scaler_dir)

    print(f"使用数据: {data_path}", flush=True)
    print(f"预测类型: {forecast_type}, 数据集: {dataset_id}", flush=True)
    print(f"使用 {historical_days} 天的历史数据用于特征提取", flush=True)
    
    # 检查是否有模式特定模型
    pattern_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))] if os.path.exists(model_dir) else []
    if not pattern_dirs:
        raise FileNotFoundError(f"没有找到模式特定的模型，请先训练 pattern_specific 模型: {model_dir}")
    print(f"找到以下模式特定模型: {pattern_dirs}", flush=True)
    
    # 加载时间序列数据
    ts_data_path = data_path
    if not os.path.exists(ts_data_path):
        raise FileNotFoundError(f"时间序列数据文件不存在: {ts_data_path}")

    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 确保我们有覆盖预测时间范围的数据
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # 检查是否有足够的历史数据
    req_history_start = start_datetime - timedelta(days=historical_days)
    if ts_data.index.min() > req_history_start:
        raise ValueError(f"没有足够的历史数据。需要从 {req_history_start} 开始的数据")
    
    # 获取预测期间的实际数据（用于评估）
    actual_data = ts_data.loc[start_datetime:end_datetime].copy()
    
    # 构建预测时间点列表
    forecast_times = []
    current_time = start_datetime
    while current_time <= end_datetime:
        forecast_times.append(current_time)
        current_time += timedelta(minutes=forecast_interval)
    
    # 准备结果容器
    results = []
    
    # 保存最近的预测和实际负荷数据，用于模式识别
    recent_loads = []
    
    # 存储已加载的模型，避免重复加载
    loaded_models = {}
    
    # 对每个时间点进行预测
    print(f"开始滚动预测，共 {len(forecast_times)} 个时间点...", flush=True)
    for i, forecast_time in enumerate(forecast_times):
        if i % 50 == 0:  # 每50次预测显示一次进度
            print(f"正在预测 {i+1}/{len(forecast_times)}: {forecast_time}", flush=True)
            
        try:
            # 准备输入数据
            X = DatasetBuilder.prepare_data_for_rolling_forecast(
                ts_data, 
                forecast_time,
                interval=forecast_interval,
                value_column=value_column
            )
            
            # 基于时间和负荷历史识别模式
            pattern = select_prediction_model(forecast_time, recent_loads)
            
            # --- 更新: 加载模型时传递正确的目录 ---
            if pattern not in loaded_models:
                try:
                    # 需要修改 load_model_for_pattern 来处理新路径
                    pattern_model_dir = f"{model_dir}/{pattern}" # 假设模式子目录
                    if not os.path.exists(pattern_model_dir):
                         print(f"模式 {pattern} 目录不存在，使用默认模型目录 {model_dir}", flush=True)
                         pattern_model_dir = model_dir # 回退
                    
                    # 假设 load_model_for_pattern 加载 PeakAwareConvTransformer
                    loaded_models[pattern] = PeakAwareConvTransformer.load(save_dir=pattern_model_dir)
                except Exception as e:
                    print(f"加载 {pattern} 模式模型失败: {e}，使用默认模型", flush=True)
                    if 'default' not in loaded_models:
                        loaded_models['default'] = PeakAwareConvTransformer.load(save_dir=model_dir)
                    pattern = 'default'
            # ---------------------------------------

            # 获取模型
            model = loaded_models[pattern]
            
            # 标准化输入数据
            X_reshaped = X.reshape(X.shape[0], -1)
            X_scaled = scaler_manager.transform('X', X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # 进行预测
            raw_pred = model.predict(X_scaled)
            
            # 确保形状正确
            raw_pred_shaped = raw_pred.reshape(-1, 1) if len(raw_pred.shape) == 1 else raw_pred
            
            # 反向标准化
            pred_inverse = scaler_manager.inverse_transform('y', raw_pred_shaped)
            predicted_value = pred_inverse.flatten()[0]
            
            # 获取实际值（如果有）
            actual_value = np.nan
            if forecast_time in actual_data.index:
                actual_value = actual_data.loc[forecast_time, value_column]
                # 更新最近负荷历史
                recent_loads.append(actual_value)
                # 只保留最近的N个点
                if len(recent_loads) > 100:
                    recent_loads = recent_loads[-100:]
            
            # 添加到结果中
            results.append({
                'datetime': forecast_time,
                'predicted': predicted_value,
                'actual': actual_value,
                'pattern': pattern
            })
            
        except Exception as e:
            print(f"预测 {forecast_time} 时出错: {e}", flush=True)
            # 继续下一个时间点的预测
    
    # 创建结果数据框
    results_df = pd.DataFrame(results)
    
    # 应用自适应平滑处理
    if len(results_df) > 0:
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].values,
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/rolling_forecast_pattern_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}", flush=True)
    
    # 计算指标（仅对有实际值的时间点）
    valid_results = results_df.dropna(subset=['actual'])
    
    if len(valid_results) > 0:
        # 计算原始预测的指标
        mae = np.mean(np.abs(valid_results['actual'] - valid_results['predicted']))
        rmse = np.sqrt(np.mean((valid_results['actual'] - valid_results['predicted']) ** 2))
        mape = np.mean(np.abs((valid_results['actual'] - valid_results['predicted']) / valid_results['actual'])) * 100
        
        # 计算平滑后预测的指标
        mae_smooth = np.mean(np.abs(valid_results['actual'] - valid_results['predicted_smoothed']))
        rmse_smooth = np.sqrt(np.mean((valid_results['actual'] - valid_results['predicted_smoothed']) ** 2))
        mape_smooth = np.mean(np.abs((valid_results['actual'] - valid_results['predicted_smoothed']) / valid_results['actual'])) * 100
        
        print(f"原始预测指标:", flush=True)
        print(f"MAE: {mae:.2f}", flush=True)
        print(f"RMSE: {rmse:.2f}", flush=True)
        print(f"MAPE: {mape:.2f}%", flush=True)
        
        print(f"\n平滑后预测指标:", flush=True)
        print(f"MAE: {mae_smooth:.2f}", flush=True)
        print(f"RMSE: {rmse_smooth:.2f}", flush=True)
        print(f"MAPE: {mape_smooth:.2f}%", flush=True)
        
        # 保存指标
        metrics = {
            'original': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            },
            'smoothed': {
                'mae': mae_smooth,
                'rmse': rmse_smooth,
                'mape': mape_smooth
            },
            'period': f"{start_date} to {end_date}",
            'forecast_interval': forecast_interval,
            'forecast_type': forecast_type
        }
        
        metrics_path = f"{results_dir}/rolling_metrics_pattern_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)

def load_model_for_pattern(pattern, model_dir="models/convtrans"):
    """
    根据识别的模式加载相应的模型
    
    参数:
    pattern (str): 模式类型
    model_dir (str): 模型目录
    
    返回:
    model: 加载的模型对象
    """
    # 检查模式特定模型文件是否存在
    pattern_model_path = f"{model_dir}/convtrans_model_{pattern}.pth"
    pattern_config_path = f"{model_dir}/{TorchConvTransformer.model_type}_config_{pattern}.json"
    pattern_shape_path = f"{model_dir}/input_shape_{pattern}.json"
    
    if os.path.exists(pattern_model_path) and os.path.exists(pattern_config_path) and os.path.exists(pattern_shape_path):
        try:
            print(f"加载 {pattern} 模式的专用模型...", flush=True)
            
            # 1. 创建临时目录
            temp_dir = f"{model_dir}/temp_load_{pattern}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 2. 复制文件到临时目录，使用标准名称
            shutil.copy2(pattern_model_path, f"{temp_dir}/{TorchConvTransformer.model_type}_model.pth")
            shutil.copy2(pattern_config_path, f"{temp_dir}/{TorchConvTransformer.model_type}_config.json")
            shutil.copy2(pattern_shape_path, f"{temp_dir}/input_shape.json")
            
            # 3. 从临时目录加载模型
            model = TorchConvTransformer.load(save_dir=temp_dir)
            
            # 4. 删除临时目录
            shutil.rmtree(temp_dir)
            
            return model
        except Exception as e:
            print(f"加载 {pattern} 模式模型失败: {e}，使用默认模型", flush=True)
    else:
        print(f"未找到 {pattern} 模式的专用模型，使用默认模型...", flush=True)
    
    # 如果无法加载特定模式模型，回退到默认模型
    return TorchConvTransformer.load(save_dir=model_dir)

def apply_adaptive_smoothing(predictions, timestamps, load_levels=None):
    """
    根据时间段和负荷水平自适应地应用平滑处理
    
    参数:
    predictions (array): 原始预测值
    timestamps (array): 对应的时间戳
    load_levels (array, optional): 对应的负荷水平，用于动态调整平滑因子
    
    返回:
    array: 平滑后的预测值
    """
    smoothed = np.copy(predictions)
    
    # 如果数据点太少，不进行平滑
    if len(predictions) < 5:
        return smoothed
    
    # 首先应用指数加权移动平均
    alpha_day = 0.5   # 日间平滑因子 (较小)
    alpha_night = 0.8  # 夜间平滑因子 (较大)
    
    for i in range(1, len(predictions)):
        # 将numpy.datetime64转换为可以获取小时的格式
        if isinstance(timestamps[i], np.datetime64):
            # 转换为pandas Timestamp对象
            ts = pd.Timestamp(timestamps[i])
            hour = ts.hour
        else:
            # 如果已经是datetime或pandas Timestamp
            hour = timestamps[i].hour
            
        # 选择合适的平滑因子
        if hour >= 22 or hour < 6:  # 夜间
            alpha = alpha_night
        else:  # 日间
            alpha = alpha_day
            
        # 应用指数平滑
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * predictions[i]
    
    # 对夜间数据点再应用中值滤波以去除毛刺
    for i in range(2, len(predictions)-2):
        # 将numpy.datetime64转换为可以获取小时的格式
        if isinstance(timestamps[i], np.datetime64):
            # 转换为pandas Timestamp对象
            ts = pd.Timestamp(timestamps[i])
            hour = ts.hour
        else:
            # 如果已经是datetime或pandas Timestamp
            hour = timestamps[i].hour
            
        if hour >= 22 or hour < 6:  # 仅对夜间数据应用
            # 使用5点中值滤波
            window = [smoothed[i-2], smoothed[i-1], smoothed[i], smoothed[i+1], smoothed[i+2]]
            smoothed[i] = np.median(window)
    
    return smoothed

def select_prediction_model(timestamp, recent_loads=None):
    """
    根据时间和最近负荷情况选择合适的预测模型
    
    参数:
    timestamp (datetime): 当前时间点
    recent_loads (array): 最近的负荷数据，用于计算波动性
    
    返回:
    str: 模型类型标识
    """
    # 将numpy.datetime64转换为可以获取小时的格式
    if isinstance(timestamp, np.datetime64):
        # 转换为pandas Timestamp对象
        ts = pd.Timestamp(timestamp)
        hour = ts.hour
    else:
        # 如果已经是datetime或pandas Timestamp
        hour = timestamp.hour
    
    # 计算负荷波动性 (如果有历史数据)
    volatility = 0
    if recent_loads is not None and len(recent_loads) > 10:
        # 使用最近10个点的标准差作为波动性指标
        volatility = np.std(recent_loads[-10:])
    
    # 夜间模式 (22:00-6:00)
    if hour >= 22 or hour < 6:
        if volatility < 50:  # 低波动阈值，需要根据实际数据调整
            return "night_stable"
        else:
            return "night_volatile"
    # 日间高峰模式 (8:00-20:00)
    elif 8 <= hour <= 20:
        return "daytime_peak"
    # 过渡时段
    else:
        return "transition"
    
def perform_combined_forecast(data_path, typical_days_path, start_date, end_date, forecast_interval=15, 
                              peak_hours=(8, 20), 
                              valley_hours=(0, 6),
                              peak_weight=5.0, 
                              valley_weight=1.5,
                              apply_smoothing=False,dataset_id='上海'):
    """使用高峰模型和非高峰模型结合典型日模式进行预测"""
    # 加载典型日信息
    ts_typical_days_path = typical_days_path
    ts_data_path = data_path
    
    try:
        typical_days_df = pd.read_csv(ts_typical_days_path)
        typical_days_df['typical_day'] = pd.to_datetime(typical_days_df['typical_day']).dt.date
        
        ts_data = pd.read_csv(ts_data_path, index_col=0)
        ts_data.index = pd.to_datetime(ts_data.index)
        
        use_typical_days = True
        print("已加载典型日信息", flush=True)
    except Exception as e:
        print(f"加载典型日信息失败: {e}", flush=True)
        use_typical_days = False
    
    # 获取模型预测
    peak_results = perform_rolling_forecast_with_peak_awareness(
        data_path=data_path,
        start_date=start_date, end_date=end_date, 
        forecast_interval=forecast_interval,
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=peak_weight,
        valley_weight=1,
        dataset_id=dataset_id
    )
    
    non_peak_results = perform_rolling_forecast_with_non_peak_awareness(
        data_path=data_path,
        start_date=start_date, end_date=end_date, 
        forecast_interval=forecast_interval,
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=1,
        valley_weight=valley_weight,
        dataset_id=dataset_id
    )
    
    # 合并结果
    combined_results = []
    for i in range(len(peak_results)):
        forecast_time = pd.to_datetime(peak_results.iloc[i]['datetime'])
        peak_pred = peak_results.iloc[i]['predicted']
        non_peak_pred = non_peak_results.iloc[i]['predicted']
        
        # 使用典型日进行动态权重调整
        if use_typical_days:
            combined_pred, weight_peak, weight_non_peak = dynamic_weight_adjustment(
                forecast_time, peak_pred, non_peak_pred, typical_days_df, ts_data
            )
        else:
            # 在默认权重策略部分使用过渡权重
            # 使用平滑过渡的权重
            weight_peak, weight_non_peak = calculate_transition_weights(
                forecast_time.hour, 
                peak_hours=peak_hours, 
                valley_hours=valley_hours
            )
            combined_pred = weight_peak * peak_pred + weight_non_peak * non_peak_pred
        
        combined_results.append({
            'datetime': forecast_time,
            'predicted': combined_pred,
            'actual': peak_results.iloc[i]['actual'],
            'peak_pred': peak_pred,
            'non_peak_pred': non_peak_pred,
            'weight_peak': weight_peak,
            'weight_non_peak': weight_non_peak
        })
    
    # 创建结果数据框并保存
    combined_results_df = pd.DataFrame(combined_results)
    
    # 根据需要应用平滑
    if apply_smoothing:
        combined_results_df['predicted_smoothed'] = apply_adaptive_smoothing(
            combined_results_df['predicted'].values,
            combined_results_df['datetime'].values
        )
    else:
        combined_results_df['predicted_smoothed'] = combined_results_df['predicted']
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = "results/combined"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = f"{results_dir}/combined_forecast_{start_date}_{end_date}_{timestamp}.csv"
    combined_results_df.to_csv(csv_path, index=False)
    
    return combined_results_df


def apply_typical_day_correction(predicted_value, timestamp, typical_days_df, ts_data):
    """根据典型日模式对预测值进行修正"""
    month = timestamp.month
    day_type = 'Weekend' if timestamp.dayofweek >= 5 else 'Workday' 
    
    # 查找对应的典型日
    match = typical_days_df[(typical_days_df['group'] == month) & 
                          (typical_days_df['day_type'] == day_type)]
    
    if len(match) == 0:
        return predicted_value  # 无典型日数据，返回原始预测
    
    typical_day = match.iloc[0]['typical_day']
    
    # 找到对应时段的典型日负荷
    typical_data = ts_data[ts_data.index.date == typical_day]
    time_filter = (typical_data.index.hour == timestamp.hour) & (typical_data.index.minute == timestamp.minute)
    
    if not any(time_filter):
        return predicted_value  # 找不到对应时间点，返回原始预测
    
    typical_load = typical_data[time_filter]['load'].iloc[0]
    
    # 获取典型日一天内的负荷分布
    day_min = typical_data['load'].min()
    day_max = typical_data['load'].max()
    
    # 判断预测值是否超出典型日的合理范围
    if predicted_value > day_max * 1.3:  # 超出典型日最高负荷的30%
        corrected_value = day_max * 1.2  # 修正为典型日最高负荷的120%
    elif predicted_value < day_min * 0.7:  # 低于典型日最低负荷的30%
        corrected_value = day_min * 0.8  # 修正为典型日最低负荷的80%
    else:
        # 根据与典型日的比例进行微调
        ratio = 0.8  # 调整因子
        corrected_value = ratio * predicted_value + (1 - ratio) * typical_load
    
    return corrected_value

def enhanced_combined_forecast(data_path, start_date, end_date, forecast_interval=15, peak_hours=(8, 20),
                               peak_weight=2.0, apply_smoothing=False):
    """整合典型日曲线和模型预测的综合预测框架"""
    # 步骤1: 获取基础模型预测
    peak_results = perform_rolling_forecast_with_peak_awareness(
        data_path=data_path,
        start_date=start_date, end_date=end_date, 
        forecast_interval=forecast_interval,
        peak_hours=peak_hours, peak_weight=peak_weight
    )
    
    non_peak_results = perform_rolling_forecast(
        data_path=data_path,
        start_date=start_date, end_date=end_date,
        forecast_interval=forecast_interval,
        apply_smoothing=apply_smoothing
    )
    
    # 步骤2: 加载典型日信息
    typical_days_df = pd.read_csv("results/typical_days/typical_days.csv")
    typical_days_df['typical_day'] = pd.to_datetime(typical_days_df['typical_day']).dt.date
    
    ts_data = pd.read_csv("data/timeseries_load_上海.csv", index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 步骤3: 整合结果
    results = []
    for i in range(len(peak_results)):
        timestamp = pd.to_datetime(peak_results.iloc[i]['datetime'])
        
        # 1. 获取基本预测值
        peak_pred = peak_results.iloc[i]['predicted']
        non_peak_pred = non_peak_results.iloc[i]['predicted']
        
        # 2. 动态权重计算
        combined_pred, weight_peak, weight_non_peak = dynamic_weight_adjustment(
            timestamp, peak_pred, non_peak_pred, typical_days_df, ts_data
        )
        
        # 3. 典型日模式修正
        corrected_pred = apply_typical_day_correction(
            combined_pred, timestamp, typical_days_df, ts_data
        )
        
        # 构建结果记录
        results.append({
            'datetime': timestamp,
            'raw_combined': combined_pred,
            'corrected_pred': corrected_pred,
            'peak_pred': peak_pred,
            'non_peak_pred': non_peak_pred,
            'actual': peak_results.iloc[i]['actual']
        })
    
    # 创建并保存结果
    results_df = pd.DataFrame(results)
    
    # 最终输出列
    results_df['predicted'] = results_df['corrected_pred']
    results_df['predicted_smoothed'] = results_df['predicted']
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"results/enhanced/enhanced_forecast_{start_date}_{end_date}_{timestamp}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    
    print(f"增强预测结果已保存至: {csv_path}", flush=True)
    return results_df

def calculate_transition_weights(hour, peak_hours=(8, 20), valley_hours=(0, 6)):
    """
    计算高峰时段和低谷时段之间的平滑过渡权重
    
    参数:
    hour: 当前小时
    peak_hours: 高峰时段起止小时 (包含边界)
    valley_hours: 低谷时段起止小时 (包含边界)
    
    返回:
    tuple: (weight_peak, weight_non_peak) - 高峰模型权重和非高峰模型权重
    """
    peak_start, peak_end = peak_hours
    valley_start, valley_end = valley_hours
    
    # 调整valley_start以处理跨日情况(如0点)
    adjusted_valley_end = valley_end
    if valley_start > valley_end:  # 跨日情况，如(22, 6)
        if hour < valley_end:  # 当前时间在凌晨
            adjusted_valley_end = valley_end  # 正常使用
        else:
            adjusted_valley_end = 24 + valley_end  # 下一天的valley_end
    
    # 确定过渡区间
    morning_transition_start = adjusted_valley_end  # 低谷结束
    morning_transition_end = peak_start            # 高峰开始
    evening_transition_start = peak_end           # 高峰结束
    evening_transition_end = 24 + valley_start if valley_start == 0 else valley_start  # 低谷开始，处理0点问题
    
    # 标准化时间以处理跨日
    adjusted_hour = hour
    if hour < valley_end and valley_start > valley_end:
        adjusted_hour = hour + 24  # 凌晨时段加24小时以便于计算
    
    # 核心高峰时段
    if peak_start <= hour <= peak_end:
        return 1.0, 0.0
    
    # 核心低谷时段
    if (valley_start <= hour <= 23 and valley_start > valley_end) or \
       (0 <= hour <= valley_end and valley_start > valley_end) or \
       (valley_start <= hour <= valley_end and valley_start <= valley_end):
        return 0.0, 1.0
    
    # 早晨过渡时段 (低谷→高峰)
    if morning_transition_start <= adjusted_hour <= morning_transition_end:
        # 计算权重比例 (0->1)
        ratio = (adjusted_hour - morning_transition_start) / (morning_transition_end - morning_transition_start)
        weight_peak = ratio
        weight_non_peak = 1 - ratio
        return weight_peak, weight_non_peak
    
    # 晚间过渡时段 (高峰→低谷)
    if evening_transition_start <= adjusted_hour <= evening_transition_end:
        # 计算权重比例 (1->0)
        ratio = (adjusted_hour - evening_transition_start) / (evening_transition_end - evening_transition_start)
        weight_peak = 1 - ratio
        weight_non_peak = ratio
        return weight_peak, weight_non_peak
    
    # 其他情况返回默认值
    return 0.5, 0.5

def apply_dynamic_scaling(predicted_values, recent_history, scaling_factor=1.2):
    """根据最近历史负荷动态调整预测值"""
    # 计算最近历史数据的峰值
    recent_peak = np.max(recent_history)
    predicted_peak = np.max(predicted_values)
    
    # 如果预测峰值低于历史峰值，进行调整
    if predicted_peak < recent_peak:
        # 计算缩放因子
        scale = min(recent_peak / predicted_peak * scaling_factor, 1.5)  # 限制最大缩放
        
        # 计算每个时间点相对于预测峰值的占比
        relative_values = predicted_values / predicted_peak
        
        # 应用缩放，保持相对关系
        adjusted_prediction = relative_values * predicted_peak * scale
        
        return adjusted_prediction
    
    return predicted_values


def perform_rolling_forecast_with_peak_awareness_and_real_time_adjustment(data_path, start_date, end_date, forecast_interval=15, 
                                             peak_hours=(8, 20), valley_hours=(0, 6),
                                             peak_weight=5.0, valley_weight=1.5,
                                             apply_smoothing=False, model_timestamp=None, dataset_id='上海', forecast_type='load'):
    """
    使用具有高峰感知能力的模型进行日内滚动预测以及实时调整
    """
    print(f"\n=== 使用高峰感知模型进行日内滚动预测和实时调整 ===", flush=True)
    print(f"预测期间: {start_date} 到 {end_date}", flush=True)
    print(f"预测间隔: {forecast_interval}分钟", flush=True)
    print(f"高峰时段: {peak_hours[0]}:00 - {peak_hours[1]}:00", flush=True)
    print(f"平滑处理: {'启用' if apply_smoothing else '禁用'}", flush=True)
    print(f"预测类型: {forecast_type}", flush=True)
    
    # 创建目录
    model_dir = f"models/convtrans_peak/{dataset_id}"
    results_dir = f"results/convtrans_peak/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)

    # 初始化缩放器管理器
    scaler_dir = f"models/scalers/convtrans_peak/{dataset_id}"
    
    # 如果指定了时间戳，加载对应的模型和缩放器
    if model_timestamp:
        model_dir = f"{model_dir}/{model_timestamp}"
        scaler_dir = f"{scaler_dir}/{model_timestamp}"
    
    # 检查模型和缩放器是否存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"缩放器目录不存在: {scaler_dir}")
    
    print(f"从 {model_dir} 加载已训练的模型...", flush=True)
    model = PeakAwareConvTransformer.load(save_dir=model_dir)
    
    print(f"从 {scaler_dir} 加载缩放器...", flush=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 加载时间序列数据
    ts_data_path = data_path
    if not os.path.exists(ts_data_path):
        raise FileNotFoundError(f"时间序列数据文件不存在: {ts_data_path}")

    print(f"从 {ts_data_path} 加载时间序列数据...", flush=True)
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据集构建器
    dataset_builder = DatasetBuilder(seq_length=int(1440/forecast_interval), pred_horizon=1, standardize=False)
    
    # 确保有足够的历史数据
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # 扩展历史数据范围（向前扩展7天）
    extended_history_start = start_datetime - timedelta(days=7)
    if ts_data.index.min() > extended_history_start:
        raise ValueError(f"没有足够的历史数据。需要从 {extended_history_start} 开始的数据")
    
    # 获取扩展后的数据范围
    extended_data = ts_data.loc[extended_history_start:end_datetime].copy()
    
    # 构建预测时间点列表
    forecast_times = []
    current_time = start_datetime
    while current_time <= end_datetime:
        forecast_times.append(current_time)
        current_time += timedelta(minutes=forecast_interval)
    
    # 准备结果容器
    results = []

    # 实时误差调整机制的变量
    actual_pred_ratios = []   # 存储实际值/预测值的比例
    error_factors = {}        # 按小时存储误差调整因子

    # 初始化每个小时的调整因子为1.0（不调整）
    for hour in range(24):
        error_factors[hour] = 1.0
    
    # 对每个时间点进行预测
    print(f"开始滚动预测，共 {len(forecast_times)} 个时间点...", flush=True)
    for i, forecast_time in enumerate(forecast_times):
        if i % 50 == 0:  # 每50次预测显示一次进度
            print(f"正在预测 {i+1}/{len(forecast_times)}: {forecast_time}", flush=True)
        
        try:
            # 准备输入数据
            enhanced_data = dataset_builder.build_dataset_with_peak_awareness(
                df=extended_data,
                date_column='datetime',
                value_column=forecast_type,
                peak_hours=peak_hours,
                valley_hours=valley_hours,
                peak_weight=peak_weight,
                valley_weight=valley_weight,
                start_date=extended_history_start,
                end_date=forecast_time
            )
            # print(f"增强数据集生成成功，形状: {enhanced_data.shape}")
            
            # 提取最近的历史数据
            X = enhanced_data.iloc[-dataset_builder.seq_length:].drop(columns=[forecast_type]).values
            # print(f"输入数据形状: {X.shape}")
            
            X = X.reshape(1, X.shape[0], X.shape[1])  # 添加批次维度
            # print(f"重塑后的输入数据形状: {X.shape}")
            
            # 标准化输入数据
            X_scaled = scaler_manager.transform('X', X.reshape(1, -1)).reshape(X.shape)
            # print(f"标准化后的输入数据形状: {X_scaled.shape}")
            
            # 预测
            raw_pred = model.predict(X_scaled)
            # print(f"模型预测成功，原始预测值: {raw_pred}")
            
            pred_inverse = scaler_manager.inverse_transform('y', raw_pred)
            predicted_value = pred_inverse.flatten()[0]
            # print(f"反标准化后的预测值: {predicted_value}")
            
            # 根据当前时间点的小时应用误差调整因子
            current_hour = forecast_time.hour
            adjusted_prediction = predicted_value * error_factors[current_hour]

            # 获取实际值（如果有）
            actual_value = np.nan
            if forecast_time in ts_data.index:
                actual_value = ts_data.loc[forecast_time, forecast_type]
                
                # 当有实际值时，更新误差调整因子
                if not np.isnan(actual_value) and predicted_value > 0:
                    # 计算当前实际值与预测值的比例
                    current_ratio = actual_value / predicted_value
                    
                    # 如果比例在合理范围内，将其添加到比例历史中
                    if 0.5 <= current_ratio <= 2.0:  # 过滤极端情况
                        actual_pred_ratios.append((forecast_time.hour, current_ratio))
                        
                        # 保留最近100个比例值
                        if len(actual_pred_ratios) > 100:
                            actual_pred_ratios.pop(0)
                        
                        # 更新当前小时的调整因子：新因子 = 0.9 * 旧因子 + 0.1 * 当前比例
                        # 这是一个指数加权移动平均，赋予最近观测更高权重
                        error_factors[current_hour] = 0.9 * error_factors[current_hour] + 0.1 * current_ratio
                        
                        # 限制调整因子在合理范围内
                        error_factors[current_hour] = max(0.7, min(error_factors[current_hour], 1.5))

            
            # 添加到结果中
            results.append({
                'datetime': forecast_time,
                'predicted_raw': predicted_value,
                'predicted': adjusted_prediction,  # 使用调整后的预测
                'actual': actual_value,
                'error_factor': error_factors[current_hour],
                'is_peak': forecast_time.hour >= peak_hours[0] and forecast_time.hour <= peak_hours[1]
            })
        
            # # 每50个预测点打印一次当前的误差调整因子
            # if i % 50 == 0 and i > 0:
            #     print(f"当前误差调整因子: {error_factors}")

        except Exception as e:
            print(f"预测 {forecast_time} 时出错: {e}", flush=True)
            # 继续下一个时间点的预测
    
    # 创建结果数据框
    results_df = pd.DataFrame(results, columns=['datetime', 'actual', 'predicted', 'is_peak'])
    
    # 应用平滑处理（如果启用）
    if apply_smoothing and len(results_df) > 0:
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].values,
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    else:
        results_df['predicted_smoothed'] = results_df['predicted']
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/peak_aware_forecast_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}", flush=True)
    
    return results_df

def perform_weather_aware_rolling_forecast(data_path, start_date, end_date, 
                                          forecast_interval=15, 
                                          peak_hours=(8, 20), valley_hours=(0, 6),
                                          peak_weight=5.0, valley_weight=1.5,
                                          apply_smoothing=False, 
                                          dataset_id='福建', 
                                          forecast_type='load',
                                          historical_days=8,
                                          weather_features=None):
    """
    执行天气感知的滚动预测
    
    Args:
        data_path: 包含天气数据的时间序列数据路径
        start_date: 预测开始日期
        end_date: 预测结束日期
        forecast_interval: 预测时间间隔（分钟）
        peak_hours: 高峰时段起止小时
        valley_hours: 低谷时段起止小时
        peak_weight: 高峰权重
        valley_weight: 低谷权重
        apply_smoothing: 是否应用平滑
        dataset_id: 数据集ID
        forecast_type: 预测类型
        historical_days: 用于模型输入的历史数据天数
        weather_features: 天气特征列表
    
    Returns:
        预测结果DataFrame
    """
    
    print(f"=== 天气感知滚动预测 ===")
    print(f"预测期间: {start_date} 到 {end_date}")
    print(f"预测间隔: {forecast_interval} 分钟")
    
    # 导入必要的模块
    from data.data_loader import DataLoader
    from data.dataset_builder import DatasetBuilder
    from models.torch_models import WeatherAwareConvTransformer
    from utils.scaler_manager import ScalerManager
    
    # 根据预测类型选择合适的目录
    value_column = forecast_type
    
    model_dir = f"models/convtrans_weather/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/convtrans_weather/{forecast_type}/{dataset_id}"
    results_dir = f"results/convtrans_weather/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"使用数据: {data_path}")
    print(f"预测类型: {forecast_type}, 数据集: {dataset_id}")
    print(f"模型目录: {model_dir}")
    print(f"缩放器目录: {scaler_dir}")
    
    # 检查模型和缩放器目录是否存在
    if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir)):
        raise FileNotFoundError(f"天气感知模型目录不存在或其中没有模型文件: {model_dir}")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"缩放器目录不存在: {scaler_dir}")
    
    # 加载天气感知模型
    print(f"从 {model_dir} 加载天气感知模型...")
    try:
        weather_model = WeatherAwareConvTransformer.load(save_dir=model_dir)
        print("天气感知模型加载成功")
    except Exception as e:
        print(f"加载天气感知模型失败: {e}")
        raise
    
    # 初始化缩放器管理器
    print(f"从 {scaler_dir} 加载缩放器...")
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 从模型配置获取seq_length
    seq_length = weather_model.config.get('seq_length', 96)
    print(f"使用seq_length={seq_length}")
    
    # 加载时间序列数据
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"时间序列数据文件不存在: {data_path}")

    print(f"从 {data_path} 加载时间序列数据...")
    ts_data = pd.read_csv(data_path)
    
    # 处理时间列
    if 'datetime' in ts_data.columns:
        ts_data['datetime'] = pd.to_datetime(ts_data['datetime'])
        ts_data = ts_data.set_index('datetime')
    elif 'timestamp' in ts_data.columns:
        ts_data['timestamp'] = pd.to_datetime(ts_data['timestamp'])
        ts_data = ts_data.set_index('timestamp')
    else:
        # 假设第一列是时间列
        ts_data.index = pd.to_datetime(ts_data.index)
    
    # 自动识别天气特征（如果未提供）
    if weather_features is None:
        # 排除已知的非天气列
        non_weather_cols = [value_column, 'PARTY_ID', 'hour', 'day_of_week', 'month', 
                           'is_weekend', 'is_peak', 'is_valley']
        weather_features = [col for col in ts_data.columns 
                           if col not in non_weather_cols and 'weather' in col.lower()]
        print(f"自动识别的天气特征: {weather_features}")
    
    # 初始化数据集构建器
    data_loader = DataLoader()
    dataset_builder = DatasetBuilder(
        data_loader=data_loader,
        seq_length=seq_length, 
        pred_horizon=1,
        standardize=False
    )
    
    # 设置预测时间范围
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
    
    # 扩展历史数据范围
    required_hist_start = start_datetime - timedelta(minutes=seq_length * forecast_interval)
    extended_history_start = required_hist_start - timedelta(days=historical_days)
    
    if ts_data.index.min() > extended_history_start:
        print(f"警告: 历史数据可能不足，最早数据点: {ts_data.index.min()}，需要: {extended_history_start}")
        if ts_data.index.min() > required_hist_start:
            raise ValueError(f"没有足够的历史数据。需要从 {required_hist_start} 开始的数据")
        extended_history_start = ts_data.index.min()
    
    print(f"使用从 {extended_history_start} 到 {end_datetime} 的数据范围")
    
    # 构建预测时间点列表
    forecast_times = []
    current_time = start_datetime
    while current_time <= end_datetime:
        forecast_times.append(current_time)
        current_time += timedelta(minutes=forecast_interval)
    
    # 准备结果容器
    results = []
    
    # 对每个时间点进行预测
    print(f"开始滚动预测，共 {len(forecast_times)} 个时间点...")
    for i, forecast_time in enumerate(forecast_times):
        if i % 100 == 0:  # 每100次预测显示一次进度
            print(f"正在预测 {i+1}/{len(forecast_times)}: {forecast_time}")
        
        try:
            # 使用与训练时一致的数据准备方法
            current_hist_end = forecast_time - timedelta(minutes=forecast_interval)
            
            # 动态确定用于特征工程的数据范围
            feature_eng_start = max(extended_history_start, current_hist_end - timedelta(days=historical_days))
            
            # 使用prepare_data_with_peak_awareness方法准备数据
            hist_data = ts_data.loc[feature_eng_start:current_hist_end].copy()
            
            if len(hist_data) < seq_length:
                raise ValueError(f"历史数据不足: {len(hist_data)} < {seq_length}")
            
            # 使用与训练时相同的数据准备方法
            X_temp, _, _, _ = dataset_builder.prepare_data_with_peak_awareness(
                ts_data=hist_data,
                test_ratio=0.0,  # 不需要验证集
                peak_hours=peak_hours,
                valley_hours=valley_hours,
                peak_weight=peak_weight,
                valley_weight=valley_weight,
                start_date=feature_eng_start,
                end_date=current_hist_end,
                value_column=value_column
            )
            
            if len(X_temp) < 1:
                raise ValueError(f"准备的数据不足以进行预测")
            
            # 取最后一个序列作为输入
            X = X_temp[-1:].copy()  # 形状: [1, seq_length, features]
            
            # 获取实际值（如果可用）
            actual_value = np.nan
            if forecast_time in ts_data.index:
                actual_value = ts_data.loc[forecast_time, value_column]
            
            # 标准化输入数据
            X_reshaped = X.reshape(X.shape[0], -1)
            X_scaled = scaler_manager.transform('X', X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # 进行预测
            raw_pred = weather_model.predict(X_scaled)
            
            # 反向标准化
            raw_pred_shaped = raw_pred.reshape(-1, 1) if len(raw_pred.shape) == 1 else raw_pred
            pred_inverse = scaler_manager.inverse_transform('y', raw_pred_shaped)
            predicted_value = pred_inverse.flatten()[0]
            
            # 添加到结果中
            results.append({
                'datetime': forecast_time,
                'predicted': predicted_value,
                'actual': actual_value,
                'is_peak': forecast_time.hour >= peak_hours[0] and forecast_time.hour <= peak_hours[1]
            })
            
        except Exception as e:
            print(f"预测 {forecast_time} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'datetime': forecast_time,
                'predicted': np.nan,
                'actual': ts_data.loc[forecast_time, value_column] if forecast_time in ts_data.index else np.nan,
                'is_peak': forecast_time.hour >= peak_hours[0] and forecast_time.hour <= peak_hours[1]
            })
    
    # 创建结果数据框
    results_df = pd.DataFrame(results)
    
    # 应用平滑处理（如果启用）
    if apply_smoothing and len(results_df) > 0 and not results_df['predicted'].isna().all():
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].fillna(method='ffill').fillna(method='bfill').values,
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    else:
        results_df['predicted_smoothed'] = results_df['predicted']
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{results_dir}/weather_aware_forecast_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}")
    
    # 计算评估指标
    valid_results = results_df.dropna(subset=['actual', 'predicted'])
    
    if len(valid_results) > 0:
        from utils.evaluator import calculate_metrics
        
        # 计算总体指标
        metrics = calculate_metrics(valid_results['actual'], valid_results['predicted'])
        print(f"\n=== 天气感知预测评估结果 ===")
        print(f"总预测点数: {len(results_df)}")
        print(f"有效预测点数: {len(valid_results)}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        # 计算峰时指标
        peak_results = valid_results[valid_results['is_peak']]
        if len(peak_results) > 0:
            peak_metrics = calculate_metrics(peak_results['actual'], peak_results['predicted'])
            print(f"\n峰时预测指标:")
            print(f"峰时预测点数: {len(peak_results)}")
            print(f"峰时MAE: {peak_metrics['mae']:.2f}")
            print(f"峰时RMSE: {peak_metrics['rmse']:.2f}")
            print(f"峰时MAPE: {peak_metrics['mape']:.2f}%")
    
    print(f"\n天气感知滚动预测完成！")
    return results_df