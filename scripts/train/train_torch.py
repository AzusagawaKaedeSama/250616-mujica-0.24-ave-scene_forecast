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
# Import project components
from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder
from utils.evaluator import ModelEvaluator, plot_peak_forecast_analysis, calculate_metrics, plot_forecast_results
from utils.scaler_manager import ScalerManager
from models.torch_models import TorchConvTransformer, PeakAwareConvTransformer
# print("--- DEBUG [train_torch.py]: Successfully imported models.torch_models ---", flush=True)
import shutil
from utils.weights_utils import get_time_based_weights
import json
sys.stdout.reconfigure(encoding='utf-8')

def train_forecast_model(data_path, train_start_date, train_end_date, forecast_type='load',
                         convtrans_config = {
                            'seq_length': 96,  # 输入序列长度
                            'pred_length': 1,                # 预测步长
                            'batch_size': 32,                # 批量大小
                            'lr': 1e-4,                      # 学习率
                            'epochs': 50,                    # 最大训练轮数
                            'patience': 10                   # 早停耐心值
                        }, 
                         retrain=True, dataset_id='上海',
                         peak_hours=(8, 20), valley_hours=(0, 6), peak_weight=2.5, valley_weight=1.5 # Added peak/valley params for consistency
                         ):
    """
    训练通用模型 (非高峰感知)
    
    参数:
    train_start_date (str): 训练开始日期
    train_end_date (str): 训练结束日期
    forecast_type (str): 预测类型，可以是'load'、'pv'或'wind'
    retrain (bool): 是否重新训练模型，即使已有训练好的模型
    dataset_id (str): 数据集ID
    peak_hours, valley_hours, peak_weight, valley_weight: 传递但不直接用于loss，主要用于数据准备
    
    返回:
    dict: 包含模型信息的字典
    """
    print(f"\n=== 训练通用模型 ({forecast_type} - {dataset_id}) ===")
    print(f"训练期间: {train_start_date} 到 {train_end_date}")
    
    # 根据预测类型确定数值列名
    value_column = forecast_type
    
    # --- 标准化路径 --- 
    model_base_name = 'convtrans' # 基础模型名称
    model_dir = os.path.join('models', model_base_name, forecast_type, dataset_id)
    scaler_dir = os.path.join('models', 'scalers', model_base_name, forecast_type, dataset_id)
    # results_dir 不再由训练函数直接创建或使用，可以移除
    # results_dir = os.path.join('results', model_base_name, forecast_type, dataset_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    # os.makedirs(results_dir, exist_ok=True)
    # -------------------------------------------
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 初始化缩放器管理器
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 1. 加载时间序列数据
    ts_data_path = data_path
    
    # 检查文件是否存在
    if not os.path.exists(ts_data_path):
        # Simple fallback for load - might need adjustment for pv/wind
        province_files = [f for f in os.listdir("data") if f.startswith(f"timeseries_{forecast_type}_") and f.endswith(".csv")]
        if province_files:
            ts_data_path = f"data/{province_files[0]}"
            print(f"未找到指定数据文件，将使用 {ts_data_path}")
        else:
            raise FileNotFoundError(f"找不到类型为 {forecast_type} 的时间序列数据文件")
    
    print(f"从 {ts_data_path} 加载时间序列数据...")
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据加载器和数据集构建器
    data_loader = DataLoader()
    dataset_builder = DatasetBuilder(
        data_loader=data_loader,
        seq_length=convtrans_config['seq_length'],
        pred_horizon=1,
        standardize=False
    )
    
    # 筛选训练和预测日期范围的数据
    train_data = ts_data.loc[train_start_date:train_end_date]

    # 创建数据集 - 使用包含高峰感知特征的数据集构建器，即使是非高峰模型
    # 这样可以保持特征维度一致，但训练时损失函数不同
    print("准备训练数据 (使用高峰感知特征工程以保证维度一致)...")
    X_train, y_train, X_val, y_val = dataset_builder.prepare_data_with_peak_awareness(
        ts_data=ts_data.loc[train_start_date:train_end_date],
        test_ratio=0.2, 
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=peak_weight, # Weights are used in feature engineering
        valley_weight=valley_weight,
        start_date=train_start_date,
        end_date=train_end_date,
        value_column=value_column
    )
    
    print(f"训练集 X 形状: {X_train.shape}, y 形状: {y_train.shape}")
    print(f"验证集 X 形状: {X_val.shape}, y 形状: {y_val.shape}")

    # --- DEBUG: 检查特征工程后的数据 ---
    # --- DEBUG: 检查特征工程后的数据类型和内容 (更详细) ---
    print("--- DEBUG: Checking data AFTER feature engineering (Detailed) ---")
    all_numeric = True
    problematic_features_info = []
    num_features = X_train.shape[2] # 获取特征数量

    # 逐个检查每个特征的数据类型
    for feature_idx in range(num_features):
        feature_slice = X_train[:, :, feature_idx]
        # 检查该特征的数据类型是否为 object 或非数字类型
        if not np.issubdtype(feature_slice.dtype, np.number):
            all_numeric = False
            sample_values = feature_slice[0, :5] # 获取第一个样本的前5个值
            problematic_features_info.append({
                'index': feature_idx,
                'dtype': feature_slice.dtype,
                'sample_values': sample_values
            })
            print(f"  - Feature {feature_idx}: Found non-numeric ({feature_slice.dtype}) data. Sample: {sample_values}")

    if all_numeric:
        print("  - All features appear numeric. Checking for NaN/Inf...")
        try:
            # 只有当所有特征都是数字时才检查 NaN/Inf
            print(f"    X_train contains NaN: {np.isnan(X_train).any()}")
            print(f"    X_train contains Inf: {np.isinf(X_train).any()}")
            # 假设 X_val 结构相同
            print(f"    X_val contains NaN: {np.isnan(X_val).any()}")
            print(f"    X_val contains Inf: {np.isinf(X_val).any()}")
        except TypeError as e:
             # 这理论上不应该发生，但作为保险
             print(f"    Error checking NaN/Inf even after dtype check: {e}")
    else:
        # 如果发现非数字特征，打印详细信息
        print("  - Found non-numeric features. Cannot proceed with NaN/Inf check or scaling.")
        print(f"  - Problematic Features Info: {problematic_features_info}")
        # 可以考虑在这里抛出错误或退出，因为数据有问题
        # raise TypeError(f"Non-numeric data found in features after engineering: {problematic_features_info}")

    print("--- DEBUG: End checking data AFTER feature engineering ---", flush=True)
    # ---------------------------------

    # 标准化数据
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
    
    # 确保 y_train_scaled 和 y_val_scaled 是一维的
    y_train_scaled_flat = y_train_scaled.flatten()
    y_val_scaled_flat = y_val_scaled.flatten()

    # --- DEBUG: Check for NaNs AFTER scaling (Detailed) ---
    print("--- DEBUG: Checking for NaNs AFTER scaling (Detailed - General Model) ---")
    nan_features_after_scaling_train = np.where(np.isnan(X_train_scaled).any(axis=(0, 1)))[0]
    if len(nan_features_after_scaling_train) > 0:
        print(f"  - NaN found in X_train_scaled features at indices: {nan_features_after_scaling_train}")
    else:
        print("  - No NaNs found in X_train_scaled features.")
        
    nan_features_after_scaling_val = np.where(np.isnan(X_val_scaled).any(axis=(0, 1)))[0]
    if len(nan_features_after_scaling_val) > 0:
        print(f"  - NaN found in X_val_scaled features at indices: {nan_features_after_scaling_val}")
    else:
        print("  - No NaNs found in X_val_scaled features.")
    print("--- DEBUG: End Checking for NaNs AFTER scaling ---")
    # ----------------------------------------------------

    n_features = X_train.shape[2] if len(X_train.shape) > 2 else 1
    print(f"模型输入特征数: {n_features}")
    
    # 创建和训练模型
    input_shape = X_train_scaled.shape[1:]
    
    model_config_train = convtrans_config.copy()
    model_config_train['use_peak_loss'] = False 
    
    # --- 修正模型初始化，确保传递 input_shape --- 
    # convtrans_model = PeakAwareConvTransformer(input_shape=input_shape, **model_config_train)
    # 假设 PeakAwareConvTransformer 继承自 TorchConvTransformer，它会处理 input_shape
    # 并且内部会调用 TorchForecaster 初始化模型
    # 这里我们只需要创建 PeakAwareConvTransformer 实例
    # 如果父类 __init__ 需要 input_shape，确保它被传递
    convtrans_model_instance = PeakAwareConvTransformer(input_shape=input_shape, **model_config_train)
    # --------------------------------------------------
    
    print(f"训练 {model_base_name} 模型...")
    
    # 进行正式训练
    print("\n开始正式训练...")
    try:
        # --- 确保 train 方法被调用在正确的实例上 --- 
        convtrans_model_instance.train(
            X_train_scaled, y_train_scaled_flat,
            X_val_scaled, y_val_scaled_flat,
            epochs=convtrans_config['epochs'],
            batch_size=convtrans_config['batch_size'],
            save_dir=model_dir # 传递正确的模型保存目录
        )
        # --- 确保 save 方法被调用在正确的实例上 --- 
        convtrans_model_instance.save(save_dir=model_dir)
        
        # 保存标准化器
        try:
            # 使用 ScalerManager 的 get_scaler 获取 scaler 对象
            if scaler_manager.has_scaler('X'):
                scaler_manager.save_scaler('X', scaler_manager.get_scaler('X'))
            if scaler_manager.has_scaler('y'):
                scaler_manager.save_scaler('y', scaler_manager.get_scaler('y'))
            print(f"模型和缩放器已保存到目录: {model_dir} 和 {scaler_dir}")
        except AttributeError:
             print("警告: 无法访问 scaler_manager 属性。请检查 ScalerManager 实现。")
        except Exception as save_err:
             print(f"保存标准化器时出错: {save_err}")
        
    except Exception as e:
        print(f"模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    return {
        'model': convtrans_model_instance # 返回模型实例
    }

def train_forecast_model_with_peak_awareness(data_path,
                                            train_start_date, 
                                            train_end_date, 
                                            forecast_type='load',
                                            peak_hours=(8, 20), 
                                            valley_hours=(0,6), 
                                            peak_weight=2.5, 
                                            valley_weight=1.5,
                                            convtrans_config = {
                                                'seq_length': 96,
                                                'pred_length': 1,
                                                'batch_size': 32,
                                                'lr': 1e-4,
                                                'epochs': 10,
                                                'patience': 10,
                                                'use_peak_loss': True
                                            }, 
                                            retrain=True, dataset_id='上海'):
    """
    训练具有高峰感知能力的模型
    
    参数:
        forecast_type (str): 预测类型 (load, pv, wind)
        ... (其他参数)
    """
    print("--- DEBUG: 进入 train_forecast_model_with_peak_awareness 函数 ---")
    print(f"\n=== 训练高峰感知模型 ({forecast_type} - {dataset_id}) ===")
    print(f"训练期间: {train_start_date} 到 {train_end_date}")
    print(f"高峰时段: {peak_hours[0]}:00 - {peak_hours[1]}:00, 权重: {peak_weight}")
    print(f"低谷时段: {valley_hours[0]}:00 - {valley_hours[1]}:00, 权重: {valley_weight}")
    
    # 根据预测类型确定数值列名
    value_column = forecast_type
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # --- 标准化路径 --- 
    model_base_name = 'convtrans_peak' # 高峰感知模型名称
    model_dir = os.path.join('models', model_base_name, forecast_type, dataset_id)
    scaler_dir = os.path.join('models', 'scalers', model_base_name, forecast_type, dataset_id)
    # results_dir 不再需要
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    # -------------------------------------------
    
    # 初始化缩放器管理器
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 加载时间序列数据
    ts_data_path = data_path
    if not os.path.exists(ts_data_path):
         # Simple fallback - might need adjustment for pv/wind
         province_files = [f for f in os.listdir("data") if f.startswith(f"timeseries_{forecast_type}_") and f.endswith(".csv")]
         if province_files:
             ts_data_path = f"data/{province_files[0]}"
             print(f"未找到指定数据文件，将使用 {ts_data_path}")
         else:
             raise FileNotFoundError(f"找不到类型为 {forecast_type} 的时间序列数据文件: {data_path}")
    
    print(f"从 {ts_data_path} 加载时间序列数据...")
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据集构建器
    dataset_builder = DatasetBuilder(seq_length=convtrans_config['seq_length'], pred_horizon=1, standardize=False)
    
    # 使用新的方法准备数据集
    print("准备训练数据 (使用高峰感知特征工程)...")
    X_train, y_train, X_val, y_val = dataset_builder.prepare_data_with_peak_awareness(
        ts_data=ts_data.loc[train_start_date:train_end_date],
        test_ratio=0.2,
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=peak_weight,
        valley_weight=valley_weight,
        start_date=train_start_date,
        end_date=train_end_date,
        value_column=value_column # 传递值列名
    )
    
    # 打印训练数据形状
    print(f"训练数据形状统计:")
    print(f"  - X_train 形状: {X_train.shape}")
    if len(X_train.shape) > 2:
        print(f"  - X_train 特征维度: {X_train.shape[2]}")
        print(f"  - X_train 序列长度: {X_train.shape[1]}")
    print(f"  - y_train 形状: {y_train.shape}")
    print(f"  - X_val 形状: {X_val.shape}")
    print(f"  - y_val 形状: {y_val.shape}")
    
    # --- DEBUG: 检查特征工程后的数据 ---
    # --- DEBUG: 检查特征工程后的数据类型和内容 (更详细) ---
    print("--- DEBUG: Checking data AFTER feature engineering (Detailed - Peak Aware) ---")
    all_numeric_train = True
    problematic_features_info_train = []
    num_features_train = X_train.shape[2] if len(X_train.shape) > 2 else 0 # 获取特征数量

    if num_features_train > 0:
        # 逐个检查每个特征的数据类型 (X_train)
        for feature_idx in range(num_features_train):
            feature_slice = X_train[:, :, feature_idx]
            # 检查该特征的数据类型是否为 object 或非数字类型
            if not np.issubdtype(feature_slice.dtype, np.number):
                all_numeric_train = False
                sample_values = feature_slice[0, :5] # 获取第一个样本的前5个值
                problematic_features_info_train.append({
                    'index': feature_idx,
                    'dtype': str(feature_slice.dtype), # 使用 str() 避免潜在的序列化问题
                    'sample_values': sample_values.tolist() # 转换为列表以便序列化
                })
                print(f"  - X_train Feature {feature_idx}: Found non-numeric ({feature_slice.dtype}) data. Sample: {sample_values}")

        if all_numeric_train:
            print("  - All X_train features appear numeric. Checking for NaN/Inf...")
            try:
                # 只有当所有特征都是数字时才检查 NaN/Inf
                print(f"    X_train contains NaN: {np.isnan(X_train).any()}")
                print(f"    X_train contains Inf: {np.isinf(X_train).any()}")
                # 假设 X_val 结构相同
                print(f"    X_val contains NaN: {np.isnan(X_val).any()}") # 同样需要检查 X_val
                print(f"    X_val contains Inf: {np.isinf(X_val).any()}")
            except TypeError as e:
                 # 这理论上不应该发生，但作为保险
                 print(f"    Error checking NaN/Inf even after dtype check: {e}")
        else:
            # 如果发现非数字特征，打印详细信息
            print("  - Found non-numeric features in X_train. Cannot proceed with NaN/Inf check or scaling.")
            print(f"  - Problematic X_train Features Info: {problematic_features_info_train}")
            # 这里可以选择抛出错误，因为数据格式有问题，模型无法处理
            # raise TypeError(f"Non-numeric data found in X_train features after engineering: {problematic_features_info_train}")
    else:
        print("  - X_train has no features to check or is not 3D.")

    print("--- DEBUG: End checking data AFTER feature engineering ---", flush=True)
    # ---------------------------------

    # 标准化数据
    if not scaler_manager.has_scaler('X') or retrain:
        print("拟合 X 标准化器...")
        X_reshape = X_train.reshape(X_train.shape[0], -1)
        print(f"  - X_train 重塑后用于标准化的形状: {X_reshape.shape}")
        scaler_manager.fit('X', X_reshape)
    
    if not scaler_manager.has_scaler('y') or retrain:
        print("拟合 y 标准化器...")
        y_train_reshaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        scaler_manager.fit('y', y_train_reshaped)
    
    # 应用标准化
    print("应用标准化...")
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_scaled = scaler_manager.transform('X', X_train_reshaped).reshape(X_train.shape)
    
    X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
    X_val_scaled = scaler_manager.transform('X', X_val_reshaped).reshape(X_val.shape)
    
    y_train_scaled = scaler_manager.transform('y', y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_manager.transform('y', y_val.reshape(-1, 1)).flatten()

    # --- DEBUG: Check for NaNs AFTER scaling (Detailed) ---
    print("--- DEBUG: Checking for NaNs AFTER scaling (Detailed - Peak Aware) ---")
    nan_features_after_scaling_train = np.where(np.isnan(X_train_scaled).any(axis=(0, 1)))[0]
    if len(nan_features_after_scaling_train) > 0:
        print(f"  - NaN found in X_train_scaled features at indices: {nan_features_after_scaling_train}")
    else:
        print("  - No NaNs found in X_train_scaled features.")
        
    nan_features_after_scaling_val = np.where(np.isnan(X_val_scaled).any(axis=(0, 1)))[0]
    if len(nan_features_after_scaling_val) > 0:
        print(f"  - NaN found in X_val_scaled features at indices: {nan_features_after_scaling_val}")
    else:
        print("  - No NaNs found in X_val_scaled features.")
    print("--- DEBUG: End Checking for NaNs AFTER scaling ---")
    # ----------------------------------------------------

    # 创建模型
    input_shape = X_train_scaled.shape[1:]
    
    model_config_train = convtrans_config.copy()
    model_config_train['use_peak_loss'] = True 
    
    # --- 修正模型初始化 --- 
    convtrans_model_instance = PeakAwareConvTransformer(input_shape=input_shape, **model_config_train)
    # --------------------
    
    print(f"训练 {model_base_name} 模型...")
    # --- 调用正确的训练方法 --- 
    convtrans_model_instance.train_with_peak_awareness(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        # 假设 PeakAwareDataset 已经包含 is_peak, is_valley 信息
        # train_is_peak=train_is_peak, # 这些需要从 prepare_data_with_peak_awareness 获取
        # val_is_peak=val_is_peak,     # 或者在 PeakAwareDataset 内部处理
        epochs=convtrans_config['epochs'],
        batch_size=convtrans_config['batch_size'],
        save_dir=model_dir
    )
    # -------------------------

    # 保存最终模型
    convtrans_model_instance.save(save_dir=model_dir)
    
    # 保存标准化器
    try:
        if scaler_manager.has_scaler('X'):
            scaler_manager.save_scaler('X', scaler_manager.get_scaler('X'))
        if scaler_manager.has_scaler('y'):
            scaler_manager.save_scaler('y', scaler_manager.get_scaler('y'))
        print(f"模型和缩放器已保存到目录: {model_dir} 和 {scaler_dir}")
    except AttributeError:
            print("警告: 无法访问 scaler_manager 属性。请检查 ScalerManager 实现。")
    except Exception as save_err:
            print(f"保存标准化器时出错: {save_err}")
    
    return {
        'model': convtrans_model_instance
    }

def train_forecast_model_with_non_peak_awareness(data_path,
                                                train_start_date, 
                                                train_end_date, 
                                                forecast_date, 
                                                peak_hours=(8, 20), 
                                                valley_hours=(0,6), 
                                                peak_weight=2.5, 
                                                valley_weight=1.5,
                                                convtrans_config = {
                                                    'seq_length': 96,
                                                    'pred_length': 1,
                                                    'batch_size': 32,
                                                    'lr': 1e-4,
                                                    'epochs': 10,
                                                    'patience': 10,
                                                    'use_peak_loss': True
                                                }, 
                                                retrain=True, dataset_id='上海'):
    """
    训练具有高峰感知能力的模型并预测负荷
    """
    print(f"\n=== 训练具有高峰感知能力的模型并进行负荷预测 ===")
    print(f"训练期间: {train_start_date} 到 {train_end_date}")
    print(f"预测日期: {forecast_date}")
    print(f"高峰时段: {peak_hours[0]}:00 - {peak_hours[1]}:00, 权重: {peak_weight}")
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 创建目录
    model_dir = f"models/convtrans_non_peak/{dataset_id}"
    results_dir = f"results/convtrans_non_peak/{dataset_id}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化缩放器管理器
    scaler_dir = f"models/scalers/convtrans_non_peak/{dataset_id}"
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 加载时间序列数据
    ts_data_path = data_path
    if not os.path.exists(ts_data_path):
        raise FileNotFoundError(f"时间序列数据文件不存在: {ts_data_path}")
    
    print(f"从 {ts_data_path} 加载时间序列数据...")
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据集构建器
    dataset_builder = DatasetBuilder(seq_length=convtrans_config['seq_length'], pred_horizon=1, standardize=False)
    
    # 使用新的方法准备数据集
    X_train, y_train, X_val, y_val = dataset_builder.prepare_data_with_peak_awareness(
        ts_data=ts_data,
        test_ratio=0.2,
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=peak_weight,
        valley_weight=valley_weight,
        start_date=train_start_date,
        end_date=train_end_date
    )
    
    # 标准化数据
    if not scaler_manager.has_scaler('X') or retrain:
        X_reshape = X_train.reshape(X_train.shape[0], -1)
        scaler_manager.fit('X', X_reshape)
    
    if not scaler_manager.has_scaler('y') or retrain:
        y_train_reshaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        scaler_manager.fit('y', y_train_reshaped)
    
    # 应用标准化
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_scaled = scaler_manager.transform('X', X_train_reshaped).reshape(X_train.shape)
    
    X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
    X_val_scaled = scaler_manager.transform('X', X_val_reshaped).reshape(X_val.shape)
    
    y_train_scaled = scaler_manager.transform('y', y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_manager.transform('y', y_val.reshape(-1, 1)).flatten()

    # 创建模型
    input_shape = X_train_scaled.shape[1:]
    convtrans_model_instance = PeakAwareConvTransformer(input_shape=input_shape, **convtrans_config)
    
    print("训练具有高峰感知能力的 ConvTransformer 模型...")
    convtrans_model_instance.train_with_peak_awareness(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        epochs=convtrans_config['epochs'],
        batch_size=convtrans_config['batch_size'],
        save_dir=model_dir
    )
    # convtrans_model.save(save_dir=model_save_dir)
    # # 保存模型


    convtrans_model_instance.save(save_dir=model_dir)

    # # scaler_manager.save_scaler(timestamp, scaler_manager.scalers)

    # print(f"模型和缩放器已保存到目录: {model_save_dir} 和 {scaler_dir}")
    return {
        'model': convtrans_model_instance
    }


def train_pattern_specific_models(data_path, train_start_date, train_end_date,
                         convtrans_config = {
                            'seq_length': 96,  # 输入序列长度
                            'pred_length': 1,                # 预测步长
                            'batch_size': 32,                # 批量大小
                            'lr': 1e-4,                      # 学习率
                            'epochs': 50,                    # 最大训练轮数
                            'patience': 10                   # 早停耐心值
                        }, retrain=False, dataset_id='上海'):
    """
    为不同的负荷模式训练专门的模型
    
    参数:
    train_start_date (str): 训练开始日期
    train_end_date (str): 训练结束日期
    retrain (bool): 是否重新训练已存在的模型
    
    返回:
    dict: 包含各模式模型和训练指标的字典
    """
    print(f"\n=== 为不同负荷模式训练专门模型 ===")
    print(f"训练期间: {train_start_date} 到 {train_end_date}")
    
    # 创建目录
    model_dir = f"models/convtrans/{dataset_id}"
    os.makedirs(model_dir, exist_ok=True)
    
    # 初始化缩放器管理器
    scaler_dir = f"models/scalers/convtrans/{dataset_id}"
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # 加载时间序列数据
    ts_data_path = data_path
    
    if os.path.exists(ts_data_path):
        print(f"从 {ts_data_path} 加载时间序列数据...")
        ts_data = pd.read_csv(ts_data_path, index_col=0)
        ts_data.index = pd.to_datetime(ts_data.index)
    else:
        raise FileNotFoundError(f"时间序列数据不存在: {ts_data_path}")
    
    # 筛选训练日期范围的数据
    train_data = ts_data.loc[train_start_date:train_end_date].copy()
    
    # 根据时间和负荷特征将数据分为不同的模式
    train_data['hour'] = train_data.index.hour
    
    # 计算负荷滚动标准差（波动性）
    train_data['volatility'] = train_data['load'].rolling(window=10).std().fillna(0)
    
    # 定义各模式的数据筛选条件
    patterns = {
        # 夜间稳定模式：22:00-06:00 且波动性低
        'night_stable': (
            ((train_data['hour'] >= 22) | (train_data['hour'] < 6)) & 
            (train_data['volatility'] < 50)
        ),
        # 夜间波动模式：22:00-06:00 且波动性高
        'night_volatile': (
            ((train_data['hour'] >= 22) | (train_data['hour'] < 6)) & 
            (train_data['volatility'] >= 50)
        ),
        # 日间高峰模式：08:00-20:00
        'daytime_peak': (
            (train_data['hour'] >= 8) & (train_data['hour'] <= 20)
        ),
        # 过渡时段：06:00-08:00 或 20:00-22:00
        'transition': (
            ((train_data['hour'] >= 6) & (train_data['hour'] < 8)) | 
            ((train_data['hour'] > 20) & (train_data['hour'] < 22))
        )
    }
    
    # 训练结果容器
    model_results = {}
    
    # 先准备全部数据用于全局标准化器
    X_train_all, y_train_all, X_val_all, y_val_all = DatasetBuilder.prepare_data_for_train(
        train_data[['load']], 
        seq_length=convtrans_config['seq_length'],
        pred_horizon=1,
        test_ratio=0.2
    )
    
    # 初始化全局标准化器（如果需要）
    if not scaler_manager.has_scaler('X'):
        print("初始化特征标准化器...")
        X_reshape = X_train_all.reshape(X_train_all.shape[0], -1)
        scaler_manager.fit('X', X_reshape)
    
    if not scaler_manager.has_scaler('y'):
        print("初始化目标标准化器...")
        y_train_shaped = y_train_all.reshape(-1, 1) if len(y_train_all.shape) == 1 else y_train_all
        scaler_manager.fit('y', y_train_shaped)
    
    # 默认模型配置
    model_config = {
        'seq_length': X_train_all.shape[2],  # 输入序列长度
        'pred_length': 1,    # 预测步长
        'batch_size': 32,    # 批量大小
        'lr': 1e-4,          # 学习率
        'epochs': 50,        # 最大训练轮数
        'patience': 10       # 早停耐心值
    }
    
    # 首先检查并训练默认模型
    default_model_path = f"{model_dir}/convtrans_model.pth"
    if os.path.exists(default_model_path) and not retrain:
        print(f"从 {default_model_path} 加载现有默认模型")
        try:
            default_model = TorchConvTransformer.load(save_dir=model_dir)
            model_results['default'] = default_model
        except Exception as e:
            print(f"加载默认模型失败: {e}, 将重新训练")
            retrain = True
    
    # 如果需要训练默认模型
    if 'default' not in model_results:
        print("训练默认 ConvTransformer 模型（全部数据）...")
        
        # 数据标准化
        X_train_reshaped = X_train_all.reshape(X_train_all.shape[0], -1)
        X_train_scaled = scaler_manager.transform('X', X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(X_train_all.shape)
        
        X_val_reshaped = X_val_all.reshape(X_val_all.shape[0], -1)
        X_val_scaled = scaler_manager.transform('X', X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(X_val_all.shape)
        
        y_train_shaped = y_train_all.reshape(-1, 1) if len(y_train_all.shape) == 1 else y_train_all
        y_train_scaled = scaler_manager.transform('y', y_train_shaped)
        
        y_val_shaped = y_val_all.reshape(-1, 1) if len(y_val_all.shape) == 1 else y_val_all
        y_val_scaled = scaler_manager.transform('y', y_val_shaped)
        
        # 创建并训练默认模型
        input_shape = X_train_scaled.shape[1:]
        default_model_instance = TorchConvTransformer(input_shape=input_shape, **model_config)
        
        default_model_instance.train(
            X_train_scaled, y_train_scaled.flatten() if len(y_train_scaled.shape) > 1 else y_train_scaled,
            X_val_scaled, y_val_scaled.flatten() if len(y_val_scaled.shape) > 1 else y_val_scaled,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            save_dir=model_dir
        )
        
        model_results['default'] = default_model_instance
    
    # 为每种模式训练专门的模型
    for pattern_name, pattern_mask in patterns.items():
        # 为当前模式从默认模型复制一个模型文件
        pattern_model_path = f"{model_dir}/convtrans_model_{pattern_name}.pth"
        
        # 检查是否已有该模式的模型且不需要重新训练
        if os.path.exists(pattern_model_path) and not retrain:
            print(f"检查到现有的 {pattern_name} 模式模型文件")
            try:
                # 尝试加载现有模型
                pattern_model = TorchConvTransformer.load(
                    save_dir=model_dir,
                    filename=f"convtrans_model_{pattern_name}.pth"
                )
                model_results[pattern_name] = pattern_model
                print(f"成功加载 {pattern_name} 模式模型")
                continue  # 已成功加载，跳过训练
            except Exception as e:
                print(f"尝试加载 {pattern_name} 模式模型时出错: {e}")
                print(f"将为 {pattern_name} 模式重新训练模型")
        
        # 提取该模式的数据
        pattern_data = train_data[pattern_mask][['load']].copy()
        
        # 检查该模式是否有足够的数据点
        if len(pattern_data) < 2000:  # 根据需要调整最小数据点要求
            print(f"模式 {pattern_name} 的数据点不足 ({len(pattern_data)} < 2000)，跳过训练专门模型")
            continue
        
        print(f"\n准备训练 {pattern_name} 模式的专门模型...")
        print(f"该模式数据点数量: {len(pattern_data)}")
        
        # 准备该模式的训练数据
        try:
            X_train_pattern, y_train_pattern, X_val_pattern, y_val_pattern = DatasetBuilder.prepare_data_for_train(
                pattern_data, 
                seq_length=convtrans_config['seq_length'],
                pred_horizon=1,
                test_ratio=0.2
            )
            
            # 检查数据集大小
            if len(X_train_pattern) < 100 or len(X_val_pattern) < 20:
                print(f"模式 {pattern_name} 的处理后训练数据不足 (训练集: {len(X_train_pattern)}, 验证集: {len(X_val_pattern)})，跳过训练")
                continue
            
            # 数据标准化（使用全局标准化器）
            X_train_reshaped = X_train_pattern.reshape(X_train_pattern.shape[0], -1)
            X_train_scaled = scaler_manager.transform('X', X_train_reshaped)
            X_train_scaled = X_train_scaled.reshape(X_train_pattern.shape)
            
            X_val_reshaped = X_val_pattern.reshape(X_val_pattern.shape[0], -1)
            X_val_scaled = scaler_manager.transform('X', X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(X_val_pattern.shape)
            
            y_train_shaped = y_train_pattern.reshape(-1, 1) if len(y_train_pattern.shape) == 1 else y_train_pattern
            y_train_scaled = scaler_manager.transform('y', y_train_shaped)
            
            y_val_shaped = y_val_pattern.reshape(-1, 1) if len(y_val_pattern.shape) == 1 else y_val_pattern
            y_val_scaled = scaler_manager.transform('y', y_val_shaped)
            
            # 为不同模式定制模型参数
            pattern_config = model_config.copy()
            if pattern_name == 'night_stable':
                # 夜间稳定模式可以使用更简单的模型，学习率更低以捕捉更平稳的模式
                pattern_config['lr'] = 5e-5
                pattern_config['patience'] = 15
            elif pattern_name == 'night_volatile':
                # 夜间波动模式需要更灵敏的模型来捕捉波动
                pattern_config['lr'] = 2e-4
            elif pattern_name == 'daytime_peak':
                # 日间高峰模式可能需要更复杂的模型来捕捉复杂模式
                pattern_config['batch_size'] = 24
                pattern_config['patience'] = 8
            
            # 创建模型
            input_shape = X_train_scaled.shape[1:]
            pattern_model_instance = TorchConvTransformer(input_shape=input_shape, **pattern_config)
            
            print(f"开始训练 {pattern_name} 模式模型...")
            
            # 训练模型
            pattern_model_instance.train(
                X_train_scaled, y_train_scaled.flatten() if len(y_train_scaled.shape) > 1 else y_train_scaled,
                X_val_scaled, y_val_scaled.flatten() if len(y_val_scaled.shape) > 1 else y_val_scaled,
                epochs=pattern_config['epochs'],
                batch_size=pattern_config['batch_size']
            )
            
            # 保存模型 - 使用正确的文件名参数
            # pattern_model.save(
            #     save_dir=model_dir, 
            #     filename=f"convtrans_model_{pattern_name}.pth"
            # )
            temp_dir = f"{model_dir}/temp_{pattern_name}"
            os.makedirs(temp_dir, exist_ok=True)
            pattern_model_instance.save(save_dir=temp_dir)

            # 2. 手动复制和重命名文件

            source_path = f"{temp_dir}/{pattern_model_instance.model_type}_model.pth"
            dest_path = f"{model_dir}/convtrans_model_{pattern_name}.pth"
            shutil.copy2(source_path, dest_path)

            # 3. 复制其他必要文件（如配置文件）
            source_config = f"{temp_dir}/{pattern_model_instance.model_type}_config.json"
            dest_config = f"{model_dir}/{pattern_model_instance.model_type}_config_{pattern_name}.json"
            shutil.copy2(source_config, dest_config)

            # 4. 复制输入形状文件
            source_shape = f"{temp_dir}/input_shape.json"
            dest_shape = f"{model_dir}/input_shape_{pattern_name}.json"
            shutil.copy2(source_shape, dest_shape)

            # 5. 可选：删除临时目录
            shutil.rmtree(temp_dir)
            
            print(f"{pattern_name} 模式模型训练完成并保存")
            model_results[pattern_name] = pattern_model_instance
            
        except Exception as e:
            print(f"训练 {pattern_name} 模式模型时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== 模式专用模型训练完成 ===")
    print(f"成功训练或加载的模式模型: {list(model_results.keys())}")
    
    return model_results

def add_typical_day_features(df, typical_days_df, ts_data):
    """添加基于典型日的特征"""
    enhanced_df = df.copy()
    
    for idx, row in enhanced_df.iterrows():
        timestamp = pd.Timestamp(idx)
        month = timestamp.month
        day_type = 'Weekend' if timestamp.dayofweek >= 5 else 'Workday'
        
        # 查找对应月份和日类型的典型日
        match = typical_days_df[(typical_days_df['group'] == month) & 
                              (typical_days_df['day_type'] == day_type)]
        
        if len(match) > 0:
            typical_day = match.iloc[0]['typical_day']
            
            # 找到相应时间点的典型日负荷值
            typical_data = ts_data[ts_data.index.date == typical_day]
            hour_data = typical_data[typical_data.index.hour == timestamp.hour]
            
            if len(hour_data) > 0:
                # 添加典型日特征
                enhanced_df.at[idx, 'typical_load'] = hour_data['load'].mean()
                # 添加与典型日的差异比例
                if 'load' in enhanced_df.columns:
                    enhanced_df.at[idx, 'typical_load_ratio'] = row['load'] / hour_data['load'].mean()
    
    # 填充缺失值
    enhanced_df = enhanced_df.fillna(method='ffill').fillna(method='bfill')
    return enhanced_df

def train_weather_aware_model(data_path, train_start_date, train_end_date, 
                             forecast_type='load',
                             weather_features=None,
                             convtrans_config = {
                                'seq_length': 96,
                                'pred_length': 1,
                                'batch_size': 32,
                                'lr': 1e-4,
                                'epochs': 50,
                                'patience': 10,
                                'use_peak_loss': True
                             }, 
                             retrain=True, dataset_id='福建',
                             peak_hours=(8, 20), valley_hours=(0, 6), 
                             peak_weight=2.5, valley_weight=1.5):
    """
    训练天气感知模型，基于现有的高精度PeakAwareConvTransformer架构
    
    参数:
    data_path (str): 包含天气数据的CSV文件路径
    train_start_date (str): 训练开始日期
    train_end_date (str): 训练结束日期
    forecast_type (str): 预测类型，默认'load'
    weather_features (list): 天气特征列表
    convtrans_config (dict): 模型配置
    retrain (bool): 是否重新训练
    dataset_id (str): 数据集ID
    peak_hours, valley_hours: 峰谷时段定义
    peak_weight, valley_weight: 峰谷权重
    
    返回:
    dict: 包含模型信息的字典
    """
    print(f"\n=== 训练天气感知模型 ({forecast_type} - {dataset_id}) ===")
    print(f"训练期间: {train_start_date} 到 {train_end_date}")
    
    # 导入天气感知模型
    from models.torch_models import WeatherAwareConvTransformer
    
    # 根据预测类型确定数值列名
    value_column = forecast_type
    
    # 标准化路径
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
    print(f"从 {data_path} 加载天气和负荷数据...")
    
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
        # 自动识别天气特征（排除已知的非天气列）
        non_weather_cols = [value_column, 'PARTY_ID', 'hour', 'day_of_week', 'month', 
                           'is_weekend', 'is_peak', 'is_valley']
        weather_features = [col for col in train_data.columns 
                           if col not in non_weather_cols and 'weather' in col.lower()]
    
    print(f"识别到的天气特征: {weather_features}")
    
    # 3. 使用现有的数据集构建器准备数据（包含峰值感知特征）
    data_loader = DataLoader()
    dataset_builder = DatasetBuilder(
        data_loader=data_loader,
        seq_length=convtrans_config['seq_length'],
        pred_horizon=1,
        standardize=False
    )
    
    print("准备训练数据（使用峰值感知特征工程 + 天气特征）...")
    
    # 使用现有的峰值感知数据准备方法
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
    model_config_train['use_peak_loss'] = True  # 启用峰值感知损失
    
    print("创建天气感知模型...")
    weather_model = WeatherAwareConvTransformer(
        input_shape=input_shape, 
        weather_features=weather_features,
        **model_config_train
    )
    
    # 6. 训练模型
    print(f"训练天气感知模型...")
    
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
        
        print("天气感知模型训练完成！")
        
    except Exception as e:
        print(f"模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    return {
        'model': weather_model,
        'weather_features': weather_features,
        'model_dir': model_dir,
        'scaler_dir': scaler_dir
    }

