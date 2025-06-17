#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练 Keras LSTM/GRU 电力负荷预测模型
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# 添加tensorflow导入
import tensorflow as tf
import sys
sys.stdout.reconfigure(encoding='utf-8')
# 导入预先定义的模型类
from models.keras_models import KerasGRU, KerasLSTM, KerasModelFactory
from utils.scaler_manager import ScalerManager
from utils.evaluator import ModelEvaluator # Keep for evaluation function
from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder

# 移除 scene_forecasting 导入尝试
# import importlib.util
# try:
#     # 尝试导入scene_forecasting中的模型
#     import sys
#     sys.path.append('.')
#     from scripts.scene_forecasting import ConvTransformerModel, load_convtransformer_model
# except ImportError:
#     print("无法导入scene_forecasting中的ConvTransformer模型，将尝试内部实现")
#     # ConvTransformer模型定义，如果无法导入原始模型
#     ConvTransformerModel = None
#     load_convtransformer_model = None

# 设置随机种子，确保结果可重现
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练 Keras LSTM/GRU 电力负荷预测模型') # Updated description
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, required=True, help='训练数据路径 (例如: data/timeseries_load_上海.csv)')
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='训练数据起始日期')
    parser.add_argument('--end_date', type=str, default='2024-12-31', help='训练数据结束日期')
    
    # 模型相关参数
    parser.add_argument('--model_type', type=str, default='lstm', 
                        choices=['lstm', 'gru'], # Removed 'ensemble', 'convtransformer'
                        help='模型类型 (lstm 或 gru)')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (默认: 50)') # Adjusted default
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比率')
    parser.add_argument('--validation_split', type=float, default=0.2, help='验证集比例')
    # Keras model train method handles early stopping based on patience in its own callbacks
    # parser.add_argument('--early_stopping', action='store_true', help='是否启用早停')
    # parser.add_argument('--patience', type=int, default=10, help='早停耐心值') 
    
    # 序列相关参数
    parser.add_argument('--seq_length', type=int, default=96, help='输入序列长度 (默认: 96, 对应24小时)') # Adjusted default
    # parser.add_argument('--forecast_horizon', type=int, default=24, help='预测长度') # Not directly used by these models' train methods
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='results/keras_train', help='模型、日志和标准化器保存目录')
    # Removed specific model/log file args, will be structured within output_dir
    # parser.add_argument('--output_model', type=str, default='models/model.h5', help='模型保存路径')
    # parser.add_argument('--output_log', type=str, default='logs/training.json', help='训练日志保存路径')
    
    # 移除 ConvTransformer 特有参数
    # parser.add_argument('--num_heads', type=int, default=4, help='ConvTransformer注意力头数量')
    # parser.add_argument('--d_model', type=int, default=128, help='ConvTransformer模型维度')
    # parser.add_argument('--d_ff', type=int, default=256, help='ConvTransformer前馈网络维度')
    # parser.add_argument('--num_layers', type=int, default=2, help='ConvTransformer编码器层数')
    
    return parser.parse_args()

def build_dataset(data_path, start_date, end_date, validation_split=0.2, seq_length=96):
    """
    使用 DataLoader 和 DatasetBuilder 构建训练和验证数据集.
    
    Args:
        data_path (str): 数据文件路径 (应指向特定区域的timeseries文件).
        start_date (str): 训练起始日期.
        end_date (str): 训练结束日期.
        validation_split (float): 验证集比例.
        seq_length (int): 输入序列长度.
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
               返回的数据是 NumPy 数组，尚未标准化.
    """
    print(f"构建训练数据集: {start_date} 到 {end_date} 使用 DatasetBuilder")
    
    try:
        # 使用项目内部的数据加载和构建工具
        # 假设 data_path 直接指向需要的 timeseries CSV 文件
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"指定的数据文件不存在: {data_path}")

        # 从 data_path 推断区域名称 (假设格式为 '.../timeseries_load_{region_name}.csv')
        region_name = "unknown"
        filename = os.path.basename(data_path)
        if filename.startswith('timeseries_load_') and filename.endswith('.csv'):
            region_name = filename.replace('timeseries_load_', '').replace('.csv', '')
        elif filename.startswith('timeseries_pv_') and filename.endswith('.csv'):
            region_name = filename.replace('timeseries_pv_', '').replace('.csv', '')
        elif filename.startswith('timeseries_wind_') and filename.endswith('.csv'):
            region_name = filename.replace('timeseries_wind_', '').replace('.csv', '')
        print(f"从文件路径推断区域/类型: {region_name}")

        # 1. 加载数据 (DataLoader 现在可能不是必需的，因为 DatasetBuilder 可以直接读CSV)
        # data_loader = DataLoader() # 可能不需要
        # data_loader.load_region_data(region_name) # 可能不需要
        # processed_data = data_loader.preprocess() # 可能不需要

        # 2. 构建数据集 (传入 data_path)
        # 注意: DatasetBuilder 的实现可能需要调整以直接接受 data_path 或需要先加载数据
        # 此处假设 DatasetBuilder 可以处理 data_path
        # TODO: 确认 DatasetBuilder 是否需要 DataLoader 实例或可以独立工作
        data_loader_dummy = None # 占位符，如果DatasetBuilder需要可以创建
        dataset_builder = DatasetBuilder(
            data_loader=data_loader_dummy, # 传递 None 或一个实例，取决于其实现
            data_path=data_path,           # 传递文件路径
            standardize=False,             # 不在此阶段标准化
            seq_length=seq_length
            )

        # 使用 build_for_date_range 或类似方法
        # 注意: build_for_date_range 可能需要原始的 DataFrame，而不是路径
        # 需要调整 DatasetBuilder 或此处的调用方式
        # 临时方案：先加载数据
        print(f"使用 pandas 加载数据: {data_path}")
        ts_data = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # 确定 value_column (load, pv, wind)
        if 'load' in ts_data.columns:
            value_column = 'load'
        elif 'pv' in ts_data.columns:
            value_column = 'pv'
        elif 'wind' in ts_data.columns:
            value_column = 'wind'
        else:
            # Fallback or raise error if value column cannot be determined
            value_column = ts_data.columns[0] 
            print(f"警告: 无法明确确定值列，使用第一列: {value_column}")

        print(f"推断的值列: {value_column}")

        # 现在调用 prepare_data (假设这是 DatasetBuilder 的正确方法)
        # 注意：这里需要适配 DatasetBuilder 的实际接口
        # 假设 prepare_data 返回 X_train, y_train, X_val, y_val
        # (或者需要调用不同的方法如 build_for_keras)

        # 示例调用（需要根据 DatasetBuilder 调整）
        X_train, y_train, X_val, y_val = dataset_builder.prepare_data_for_train(
            ts_data=ts_data[[value_column]], # 传入包含目标列的DataFrame
            seq_length=seq_length,
            pred_horizon=1, # 假设预测步长为1
            test_ratio=validation_split,
            target_col=value_column # 明确指定目标列
        )


        # # 获取时间戳 - 可能不再需要，因为不直接生成
        # train_start_dt = pd.to_datetime(start_date)
        # timestamps = pd.date_range(train_start_dt, periods=len(y_train) + len(y_val), freq='15min')
        
        print(f"使用DatasetBuilder构建数据集完成! 形状: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")
        
        # 确保返回的是 NumPy 数组
        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

    except Exception as e:
        print(f"使用 DatasetBuilder 构建数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        # 可以选择抛出异常或返回 None
        raise # Re-raise the exception to stop execution


# 移除 TrainingLogger 类
# class TrainingLogger(object):
#     ... (代码已移除) ...

# 移除 build_convtransformer_model 函数
# def build_convtransformer_model(...):
#    ... (代码已移除) ...

# 移除 train_model 函数 (训练逻辑移至 main)
# def train_model(...):
#    ... (代码已移除) ...

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能 (使用标准化数据)
    
    Args:
        model: 训练好的 Keras 模型
        X_test: 标准化后的测试输入
        y_test: 标准化后的测试标签
    
    Returns:
        dict: 评估指标 (基于标准化值)
    """
    print("\n评估模型性能 (基于标准化值)...")
    
    # 生成预测 (预测值也是标准化的)
    predictions_scaled = model.predict(X_test)
    
    # 如果是一维的，转为二维方便计算
    if len(predictions_scaled.shape) == 1:
        predictions_scaled = predictions_scaled.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test_reshaped = y_test.reshape(-1, 1)
    else:
        y_test_reshaped = y_test

    # 计算指标 (在标准化空间计算)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_test_reshaped, predictions_scaled)
    rmse = np.sqrt(mean_squared_error(y_test_reshaped, predictions_scaled))
    # MAPE 在标准化空间计算可能意义不大，尤其是当 y_test_scaled 接近0时
    # mape = np.mean(np.abs((y_test_reshaped - predictions_scaled) / y_test_reshaped)) * 100

    print(f"验证集 MAE (标准化): {mae:.4f}")
    print(f"验证集 RMSE (标准化): {rmse:.4f}")
    # print(f"测试集 MAPE (标准化): {mape:.2f}%") # 移除或谨慎使用
    
    return {
        'mae_scaled': mae,
        'rmse_scaled': rmse,
        # 'mape_scaled': mape
    }

# 移除 visualize_results 函数 (Keras history object 可以用于绘图，或在评估后单独绘制)
# def visualize_results(...):
#    ... (代码已移除) ...

def main():
    """主函数"""
    args = parse_args()
    
    # --- 从 data_path 推断 forecast_type 和 dataset_id --- 
    data_filename = os.path.basename(args.data_path)
    forecast_type = "unknown"
    dataset_id = "unknown_region"
    
    if data_filename.startswith('timeseries_load_') and data_filename.endswith('.csv'):
        forecast_type = 'load'
        dataset_id = data_filename.replace('timeseries_load_', '').replace('.csv', '')
    elif data_filename.startswith('timeseries_pv_') and data_filename.endswith('.csv'):
        forecast_type = 'pv'
        dataset_id = data_filename.replace('timeseries_pv_', '').replace('.csv', '')
    elif data_filename.startswith('timeseries_wind_') and data_filename.endswith('.csv'):
        forecast_type = 'wind'
        dataset_id = data_filename.replace('timeseries_wind_', '').replace('.csv', '')
    else:
        print(f"警告: 无法从数据路径 {args.data_path} 推断预测类型和区域ID。将使用默认值。")
        # 可以选择在这里退出或使用默认值

    print(f"推断信息: forecast_type='{forecast_type}', dataset_id='{dataset_id}'")
    # ------------------------------------------------------

    # --- 创建符合新结构的输出目录 --- 
    base_output_path = args.output_dir # 使用用户指定的输出基路径
    model_dir = os.path.join(base_output_path, 'models', args.model_type, forecast_type, dataset_id)
    scaler_dir = os.path.join(base_output_path, 'models', 'scalers', args.model_type, forecast_type, dataset_id)
    log_dir = os.path.join(base_output_path, 'results', 'logs', args.model_type, forecast_type, dataset_id)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    # -----------------------------------------------------

    # 1. 构建数据集 (获取未标准化的数据)
    X_train, y_train, X_val, y_val = build_dataset(
        args.data_path, 
        args.start_date, 
        args.end_date,
        args.validation_split,
        args.seq_length
    )
    
    # 2. 创建 ScalerManager 并进行数据标准化
    scaler_manager = ScalerManager(scaler_dir) # 使用新的 scaler_dir
    
    print("标准化训练和验证数据...")
    
    # --- 标准化 X 数据 --- 
    if len(X_train.shape) != 3:
        raise ValueError(f"X_train 维度不正确，期望 3D (samples, timesteps, features)，得到 {X_train.shape}")
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    if not scaler_manager.has_scaler('X'):
        print("拟合 X 标准化器...")
        scaler_manager.fit('X', X_train_flat)
        # scaler_manager.save_scaler('X', scaler_manager.get_scaler('X')) # fit 内部会调用 save
    X_train_scaled_flat = scaler_manager.transform('X', X_train_flat)
    X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
    
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_val_scaled_flat = scaler_manager.transform('X', X_val_flat)
    X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
    # -----------------------
    
    # --- 标准化 y 数据 --- 
    y_train_2d = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
    if not scaler_manager.has_scaler('y'):
        print("拟合 y 标准化器...")
        scaler_manager.fit('y', y_train_2d)
        # scaler_manager.save_scaler('y', scaler_manager.get_scaler('y')) # fit 内部会调用 save
    y_train_scaled = scaler_manager.transform('y', y_train_2d)
    
    y_val_2d = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
    y_val_scaled = scaler_manager.transform('y', y_val_2d)
    # -----------------------

    print(f"标准化后数据形状: X_train={X_train_scaled.shape}, y_train={y_train_scaled.shape}, X_val={X_val_scaled.shape}, y_val={y_val_scaled.shape}")
    
    # 3. 创建并训练模型
    print(f"开始训练 {args.model_type} 模型...")
    start_time = time.time()

    input_shape = X_train_scaled.shape[1:] 
    model = KerasModelFactory.create_model(
        model_type=args.model_type,
        input_shape=input_shape,
        learning_rate=args.learning_rate,
        dropout=args.dropout
    )

    # 模型训练 (传入新的 model_dir)
    history = model.train(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=model_dir # 使用新的 model_dir
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"模型训练完成! 总耗时: {training_time:.2f} 秒")

    # 4. 保存最终模型状态 (传入新的 model_dir)
    print("保存最终模型状态...")
    model.save(save_dir=model_dir) # 使用新的 model_dir

    # 5. 评估模型 (在标准化数据上)
    metrics = evaluate_model(model, X_val_scaled, y_val_scaled)

    # 6. 保存训练日志信息 (到新的 log_path)
    log_data = {
        'args': vars(args),
        'training_time_seconds': training_time,
        'final_metrics_scaled': metrics,
        'history': {k: [float(vi) for vi in v] for k, v in history.history.items()} # 转换 numpy floats
    }
    try:
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f"训练日志已保存到: {log_path}")
    except TypeError as e:
        print(f"保存训练日志 JSON 时发生错误: {e}")
        # 可以在这里添加更详细的错误处理或尝试保存部分日志


if __name__ == "__main__":
    main() 