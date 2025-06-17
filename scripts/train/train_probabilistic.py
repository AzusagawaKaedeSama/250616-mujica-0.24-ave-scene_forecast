# data/train_probabilistic.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import necessary components from your project structure
from data.dataset_builder import DatasetBuilder
from utils.scaler_manager import ScalerManager
from models.torch_models import ProbabilisticConvTransformer # Import the new model
import sys
sys.stdout.reconfigure(encoding='utf-8')

def train_probabilistic_model(
    data_path,
    train_start_date,
    train_end_date,
    convtrans_config,
    retrain=False,
    dataset_id='default',
    forecast_type='load',
    quantiles=[0.1, 0.5, 0.9]
):
    """
    训练概率预测模型 (ProbabilisticConvTransformer)

    Args:
        data_path (str): 时间序列数据路径.
        train_start_date (str): 训练开始日期 'YYYY-MM-DD'.
        train_end_date (str): 训练结束日期 'YYYY-MM-DD'.
        convtrans_config (dict): 模型和训练参数配置.
        retrain (bool): 是否强制重新训练.
        dataset_id (str): 数据集标识符 (例如省份).
        forecast_type (str): 预测类型 ('load', 'pv', 'wind').
        quantiles (list): 需要预测的分位数列表.
    """
    print(f"\n=== 开始概率模型训练 ({forecast_type.upper()}) - {dataset_id} ===")
    print(f"Quantiles: {quantiles}")

    # 1. 设置目录路径
    model_base_name = ProbabilisticConvTransformer.model_type # 'prob_convtrans'
    model_dir = f"models/{model_base_name}/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/{model_base_name}/{forecast_type}/{dataset_id}"
    results_dir = f"results/{model_base_name}/{forecast_type}/{dataset_id}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. 检查是否需要训练
    model_path = os.path.join(model_dir, f'{model_base_name}_model.pth')
    if os.path.exists(model_path) and not retrain:
        print(f"模型已存在于 {model_path}，跳过训练。使用 --retrain 参数强制重新训练。")
        return

    # 3. 加载和准备数据
    print("加载和准备数据...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    value_column = forecast_type

    # 使用 DatasetBuilder (假设它也适用于概率模型的基本特征工程)
    seq_length = convtrans_config.get('seq_length', 96)
    interval_minutes = int(1440 / seq_length) if seq_length > 0 else 15
    pred_horizon = 1 # Quantile loss typically predicts one step ahead

    dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=pred_horizon)

    # 注意: DatasetBuilder 可能需要调整以适应概率模型 (例如，目标y可能只需要一个值)
    # 这里我们假设它能返回 X 和 y (y是单一目标值)
    # Peak awareness features might still be useful inputs
    peak_aware_config = {k: v for k, v in convtrans_config.items() if k in ['peak_hours', 'valley_hours', 'peak_weight', 'valley_weight']}
    enhanced_df = dataset_builder.build_dataset_with_peak_awareness(
        df=df, date_column=None, value_column=value_column,
        interval=interval_minutes, **peak_aware_config
    )

    # 分割数据
    train_df = enhanced_df[train_start_date:train_end_date]
    # 需要验证集，可以从训练集中划分或使用固定范围
    val_end_date = pd.to_datetime(train_end_date)
    val_start_date = (val_end_date - pd.Timedelta(days=7)).strftime('%Y-%m-%d') # Use last week for validation
    train_df_actual = train_df[:val_start_date] # Adjust training data
    val_df = train_df[val_start_date:]

    if train_df_actual.empty or val_df.empty:
        raise ValueError("训练或验证数据不足，请检查日期范围和数据。")

    # 特征和目标
    feature_columns = [col for col in enhanced_df.columns if col != value_column]
    X_train = train_df_actual[feature_columns].values
    y_train = train_df_actual[value_column].values
    X_val = val_df[feature_columns].values
    y_val = val_df[value_column].values

    # 4. 标准化数据
    print("标准化数据...")
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    X_train_scaled = scaler_manager.fit_transform('X', X_train)
    X_val_scaled = scaler_manager.transform('X', X_val)
    y_train_scaled = scaler_manager.fit_transform('y', y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_manager.transform('y', y_val.reshape(-1, 1)).flatten()

    # Reshape X for the model (batch, seq_len, features)
    num_features = len(feature_columns)
    X_train_final = X_train_scaled.reshape(-1, seq_length, num_features)
    X_val_final = X_val_scaled.reshape(-1, seq_length, num_features)

    # y needs shape (batch, 1) for quantile loss
    y_train_final = y_train_scaled.reshape(-1, 1)
    y_val_final = y_val_scaled.reshape(-1, 1)

    print(f"训练数据形状: X={X_train_final.shape}, y={y_train_final.shape}")
    print(f"验证数据形状: X={X_val_final.shape}, y={y_val_final.shape}")

    # 5. 初始化和训练模型
    input_shape = (X_train_final.shape[1], X_train_final.shape[2])
    
    # --- 使用 ProbabilisticConvTransformer --- 
    prob_model = ProbabilisticConvTransformer(
        input_shape=input_shape, 
        quantiles=quantiles, 
        **convtrans_config
    )

    print("开始训练概率模型...")
    history = prob_model.train_probabilistic(
        X_train=X_train_final,
        y_train=y_train_final,
        X_val=X_val_final,
        y_val=y_val_final,
        epochs=convtrans_config.get('epochs', 50),
        batch_size=convtrans_config.get('batch_size', 32),
        save_dir=model_dir
    )

    # 6. 保存标准化器
    scaler_manager.save_scalers()
    print(f"标准化器已保存到 {scaler_dir}")

    print(f"概率模型训练完成: {dataset_id} ({forecast_type}) - 模型保存在 {model_dir}")

# Example usage (if run directly)
if __name__ == '__main__':
    # This part is for testing the script directly
    # In practice, it will be called by scene_forecasting.py
    config_example = {
        'seq_length': 96,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 2, # Use a small number for testing
        'patience': 5,
        'peak_hours': (7, 22),
        'valley_hours': (0, 6),
        'peak_weight': 10.0, # Still potentially useful for input features
        'valley_weight': 1.5
    }
    quantiles_example = [0.1, 0.5, 0.9]
    
    try:
        train_probabilistic_model(
            data_path='data/timeseries_load_上海.csv', # Example path
            train_start_date='2024-01-01',
            train_end_date='2024-03-31',
            convtrans_config=config_example,
            retrain=True, # Force retrain for test
            dataset_id='上海',
            forecast_type='load',
            quantiles=quantiles_example
        )
    except FileNotFoundError:
        print("错误：测试需要示例数据文件 data/timeseries_load_上海.csv")
    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc() 