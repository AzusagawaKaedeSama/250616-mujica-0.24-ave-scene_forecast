#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练基于误差分布的区间预测模型
使用IntervalPeakAwareConvTransformer模型
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import pickle
import sys
sys.stdout.reconfigure(encoding='utf-8')
# 导入自定义模型和工具
from models.torch_models import IntervalPeakAwareConvTransformer
from utils.interval_forecast_utils import DataPipeline
from utils.interval_forecast_utils import preprocess_data_for_training
from utils.scaler_manager import ScalerManager
from utils.interval_forecast_utils import is_peak_hour, is_valley_hour, is_workday
from utils.interval_forecast_utils import (
    PEAK_HOURS, VALLEY_HOURS, FEATURES_CONFIG, TARGET_COLUMNS,
    TRAIN_RATIO, VALID_RATIO
)

def train_interval_model(args):
    """训练区间预测模型"""
    print(f"开始训练区间预测模型: 省份={args.province}, 预测类型={args.forecast_type}")
    
    # 确定数据日期范围
    end_date = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else datetime.now()
    if args.days:
        start_date = end_date - timedelta(days=int(args.days))
        start_date_str = start_date.strftime("%Y%m%d")
    else:
        start_date_str = args.start_date
    
    end_date_str = end_date.strftime("%Y%m%d")
    print(f"使用数据范围: {start_date_str} 至 {end_date_str}")
    
    # 创建数据管道
    data_pipeline = DataPipeline(args.province, args.forecast_type)
    
    # 获取训练数据
    print("正在加载训练数据...")
    raw_data = data_pipeline.get_data_for_range(
        start_date_str, end_date_str, 
        include_temporal_features=True
    )
    
    if raw_data is None or len(raw_data) == 0:
        print(f"错误: 无法获取足够的训练数据")
        return False
    
    print(f"成功加载原始数据: {len(raw_data)}行 x {len(raw_data.columns)}列")
    
    # 数据预处理
    print("开始数据预处理...")
    target_col = args.forecast_type
    time_col = raw_data.index.name or 'datetime'
    
    # 确保数据按时间排序
    if not raw_data.index.is_monotonic_increasing:
        raw_data = raw_data.sort_index()
    
    # 添加峰谷标记
    raw_data['is_peak'] = raw_data.index.hour.map(is_peak_hour)
    raw_data['is_valley'] = raw_data.index.hour.map(is_valley_hour)
    raw_data['is_workday'] = raw_data.index.map(is_workday)
    
    # 打印数据统计信息
    print(f"数据总量: {len(raw_data)}行")
    print(f"峰时段样本数: {raw_data['is_peak'].sum()}行")
    print(f"谷时段样本数: {raw_data['is_valley'].sum()}行")
    print(f"工作日样本数: {raw_data['is_workday'].sum()}行")
    
    # 数据预处理用于训练
    print("准备训练数据...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data_for_training(
        raw_data, 
        target_column=target_col,
        features_config=FEATURES_CONFIG[target_col],
        train_ratio=TRAIN_RATIO,
        val_ratio=VALID_RATIO,
        allow_gaps=True
    )
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 设置模型保存路径
    model_dir = os.path.join("models", "convtrans_peak", target_col, args.province)
    os.makedirs(model_dir, exist_ok=True)
    
    # 获取峰谷标记
    train_indices = X_train.index
    val_indices = X_val.index
    
    # Safely get peak/valley info using indices
    train_is_peak = raw_data.loc[train_indices, 'is_peak'].values if 'is_peak' in raw_data.columns else None
    val_is_peak = raw_data.loc[val_indices, 'is_peak'].values if 'is_peak' in raw_data.columns else None
    train_is_valley = raw_data.loc[train_indices, 'is_valley'].values if 'is_valley' in raw_data.columns else None
    val_is_valley = raw_data.loc[val_indices, 'is_valley'].values if 'is_valley' in raw_data.columns else None
    
    # 初始化并训练模型
    input_shape = X_train.shape[1:]
    print(f"创建模型，输入特征维度: {input_shape}")
    
    # 初始化模型
    model = IntervalPeakAwareConvTransformer(
        input_shape=input_shape,
        quantiles=[0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975],  # 设置分位数
        seq_length=model.config.get('seq_length', 96),
        pred_length=model.config.get('pred_length', 96),
        epochs=args.epochs or model.config.get('epochs', 100),
        batch_size=args.batch_size or model.config.get('batch_size', 32),
        lr=args.learning_rate or model.config.get('lr', 0.001),
        patience=model.config.get('patience', 10)
    )
    
    # 创建缩放器管理器
    scaler_dir = os.path.join("models", "scalers", "convtrans_peak", target_col, args.province)
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    
    # Fit scalers if they don't exist or if retraining is forced (though retrain isn't an arg here)
    if not scaler_manager.has_scaler('X'):
        print("拟合 X 标准化器...")
        scaler_manager.fit('X', X_train)
    if not scaler_manager.has_scaler('y'):
        print("拟合 y 标准化器...")
        scaler_manager.fit('y', y_train)
    
    # Transform data
    X_train_scaled = scaler_manager.transform('X', X_train)
    y_train_scaled = scaler_manager.transform('y', y_train)
    X_val_scaled = scaler_manager.transform('X', X_val)
    y_val_scaled = scaler_manager.transform('y', y_val)
    
    # 保存缩放器
    scaler_manager.save_all()
    print(f"缩放器已保存到 {scaler_dir}")
    
    # 训练模型
    print(f"开始训练模型，使用GPU: {args.use_gpu}")
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU
    
    print("使用train_with_error_capturing方法训练模型，同时捕获预测误差...")
    training_results = model.train_with_error_capturing(
        X_train_scaled, y_train_scaled, 
        X_val_scaled, y_val_scaled,
        train_is_peak=train_is_peak,
        val_is_peak=val_is_peak,
        train_is_valley=train_is_valley,
        val_is_valley=val_is_valley,
        epochs=args.epochs or 100,
        batch_size=args.batch_size or 32,
        save_dir=model_dir
    )
    
    # 保存训练配置
    config = {
        'province': args.province,
        'forecast_type': args.forecast_type,
        'data_start_date': start_date_str,
        'data_end_date': end_date_str,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_params': model.config,
        'peak_hours': PEAK_HOURS,
        'valley_hours': VALLEY_HOURS,
        'features_used': list(X_train.columns)
    }
    
    # 保存配置
    with open(os.path.join(model_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"模型训练完成并保存到 {model_dir}")
    
    # 在测试集上评估模型
    if X_test is not None and len(X_test) > 0:
        print("在测试集上评估模型...")
        X_test_scaled = scaler_manager.transform('X', X_test)
        
        # 获取测试集的峰谷标记
        test_is_peak = raw_data.loc[X_test.index, 'is_peak'].values if 'is_peak' in raw_data.columns else None
        test_is_valley = raw_data.loc[X_test.index, 'is_valley'].values if 'is_valley' in raw_data.columns else None
        
        # 执行区间预测
        predictions = model.predict_interval(X_test_scaled, is_peak=test_is_peak, is_valley=test_is_valley)
        
        # 反缩放预测结果
        for q, pred_values in predictions.items():
            predictions[q] = scaler_manager.inverse_transform('y', pred_values)
        
        # 评估中位数预测结果
        point_pred = predictions.get('p50', None)
        if point_pred is not None:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # 计算评估指标
            mae = mean_absolute_error(y_test, point_pred)
            rmse = np.sqrt(mean_squared_error(y_test, point_pred))
            r2 = r2_score(y_test, point_pred)
            
            # 计算MAPE和SMAPE
            mape = np.mean(np.abs((y_test - point_pred) / (y_test + 1e-10))) * 100
            smape = np.mean(2 * np.abs(y_test - point_pred) / (np.abs(y_test) + np.abs(point_pred) + 1e-10)) * 100
            
            print(f"测试集评估结果:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAPE: {mape:.4f}%")
            print(f"SMAPE: {smape:.4f}%")
            print(f"R²: {r2:.4f}")
            
            # 计算区间覆盖率
            if 'p10' in predictions and 'p90' in predictions:
                coverage_80 = np.mean((y_test >= predictions['p10']) & (y_test <= predictions['p90'])) * 100
                print(f"80%区间覆盖率: {coverage_80:.2f}%")
                
            if 'p5' in predictions and 'p95' in predictions:
                coverage_90 = np.mean((y_test >= predictions['p5']) & (y_test <= predictions['p95'])) * 100
                print(f"90%区间覆盖率: {coverage_90:.2f}%")
                
            if 'p2' in predictions and 'p97' in predictions:
                coverage_95 = np.mean((y_test >= predictions['p2']) & (y_test <= predictions['p97'])) * 100
                print(f"95%区间覆盖率: {coverage_95:.2f}%")
            
            # 保存评估结果
            eval_results = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'smape': float(smape),
                'r2': float(r2)
            }
            
            # 添加区间覆盖率
            if 'p10' in predictions and 'p90' in predictions:
                eval_results['coverage_80'] = float(coverage_80)
            if 'p5' in predictions and 'p95' in predictions:
                eval_results['coverage_90'] = float(coverage_90)
            if 'p2' in predictions and 'p97' in predictions:
                eval_results['coverage_95'] = float(coverage_95)
            
            # 保存评估结果
            with open(os.path.join(model_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            # 可视化预测结果
            plt.figure(figsize=(14, 8))
            
            # 绘制实际值
            plt.plot(y_test.index, y_test, 'k-', label='实际值', linewidth=2)
            
            # 绘制中位数预测
            plt.plot(y_test.index, point_pred, 'b--', label='中位数预测', linewidth=1.5)
            
            # 绘制95%置信区间
            if 'p2' in predictions and 'p97' in predictions:
                plt.fill_between(
                    y_test.index, 
                    predictions['p2'], 
                    predictions['p97'], 
                    color='lightblue', alpha=0.3, 
                    label='95%预测区间'
                )
            
            # 绘制80%置信区间
            if 'p10' in predictions and 'p90' in predictions:
                plt.fill_between(
                    y_test.index, 
                    predictions['p10'], 
                    predictions['p90'], 
                    color='blue', alpha=0.3, 
                    label='80%预测区间'
                )
            
            # 高亮峰谷时段
            peak_mask = test_is_peak
            valley_mask = test_is_valley
            
            if np.any(peak_mask):
                plt.scatter(
                    y_test.index[peak_mask], 
                    y_test[peak_mask], 
                    color='red', s=20, alpha=0.7, label='峰时段'
                )
            
            if np.any(valley_mask):
                plt.scatter(
                    y_test.index[valley_mask], 
                    y_test[valley_mask], 
                    color='green', s=20, alpha=0.7, label='谷时段'
                )
            
            plt.title(f"{args.province} - {args.forecast_type} 区间预测测试集结果")
            plt.xlabel('时间')
            plt.ylabel(args.forecast_type)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 旋转x轴日期标签以增加可读性
            plt.xticks(rotation=45)
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'test_predictions.png'))
            print(f"测试集预测可视化结果已保存")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='训练区间预测模型')
    parser.add_argument('--forecast_type', choices=['load', 'pv', 'wind'], default='load',
                        help='预测目标类型: 用电负荷(load), 光伏发电(pv), 风力发电(wind)')
    parser.add_argument('--province', required=True, help='省份名称')
    parser.add_argument('--start_date', help='训练数据开始日期，格式YYYYMMDD')
    parser.add_argument('--end_date', help='训练数据结束日期，格式YYYYMMDD，默认为当天')
    parser.add_argument('--days', type=int, help='使用过去多少天的数据，与start_date二选一')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU训练')

    args = parser.parse_args()
    
    # 验证参数
    if not args.start_date and not args.days:
        print("错误: 必须指定start_date或days参数之一")
        return 1
    
    # 开始训练
    success = train_interval_model(args)
    
    # 返回结果
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 