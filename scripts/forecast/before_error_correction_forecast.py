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
warnings.filterwarnings('ignore')
import torch
from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder
from utils.evaluator import ModelEvaluator, plot_peak_forecast_analysis, calculate_metrics, plot_forecast_results
from utils.scaler_manager import ScalerManager
from models.torch_models import TorchConvTransformer, PeakAwareConvTransformer
import shutil
from utils.weights_utils import get_time_based_weights, dynamic_weight_adjustment
import json
from scripts.forecast.forecast import perform_rolling_forecast_with_peak_awareness
import sys
sys.stdout.reconfigure(encoding='utf-8')

class AdaptivePIDController:
    """自适应PID控制器类，具有参数自动调整功能"""
    def __init__(self, kp=0.8, ki=0.1, kd=0.2, dt=1.0, max_output=1.5, min_output=0.5,
                 adaptation_rate=0.01, stability_threshold=0.05):
        """
        初始化自适应PID控制器
        
        参数:
        kp, ki, kd: PID参数
        dt: 采样时间间隔
        max_output, min_output: 输出限制
        adaptation_rate: 参数自适应速率
        stability_threshold: 稳定性阈值
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_output = max_output
        self.min_output = min_output
        self.adaptation_rate = adaptation_rate
        self.stability_threshold = stability_threshold
        
        # 状态变量
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 1.0
        
        # 自适应参数
        self.error_history = []
        self.output_history = []
        self.max_history_size = 20
        
    def compute(self, error):
        """计算PID控制输出"""
        # 积分项（带抗饱和）
        self.integral += error * self.dt
        self.integral = max(-1.0, min(1.0, self.integral))
        
        # 微分项
        derivative = (error - self.previous_error) / self.dt if self.dt > 0 else 0
        
        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 限制输出范围
        output = max(self.min_output, min(self.max_output, output))
        
        # 更新历史
        self.error_history.append(error)
        self.output_history.append(output)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
            self.output_history.pop(0)
        
        # 自适应调整参数
        self._adapt_parameters()
        
        # 更新状态
        self.previous_error = error
        self.previous_output = output
        
        return output
    
    def _adapt_parameters(self):
        """根据历史性能自适应调整PID参数"""
        if len(self.error_history) < 10:
            return
        
        # 计算误差变化率
        recent_errors = self.error_history[-10:]
        error_variance = np.var(recent_errors)
        error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        # 根据误差特征调整参数
        if error_variance > self.stability_threshold:
            # 系统不稳定，减小增益
            self.kp *= (1 - self.adaptation_rate)
            self.ki *= (1 - self.adaptation_rate)
        elif abs(error_trend) > self.stability_threshold:
            # 存在稳态误差，增加积分作用
            self.ki *= (1 + self.adaptation_rate)
        else:
            # 系统稳定，轻微增加增益以提高响应速度
            self.kp *= (1 + self.adaptation_rate * 0.5)
        
        # 确保参数在合理范围内
        self.kp = max(0.1, min(2.0, self.kp))
        self.ki = max(0.01, min(0.5, self.ki))
        self.kd = max(0.01, min(0.5, self.kd))

class SlidingWindowErrorAnalyzer:
    """滑动窗口误差分析器"""
    def __init__(self, window_size_hours=72):
        """
        初始化误差分析器
        
        参数:
        window_size_hours: 滑动窗口大小（小时）
        """
        self.window_size_hours = window_size_hours
        self.error_data = pd.DataFrame()
        
    def update(self, timestamp, actual, predicted, hour):
        """更新误差数据"""
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'actual': [actual],
            'predicted': [predicted],
            'hour': [hour],
            'error_ratio': [actual / predicted if predicted > 0 else 1.0]
        })
        self.error_data = pd.concat([self.error_data, new_row], ignore_index=True)
        
        # 保持窗口大小
        cutoff_time = timestamp - timedelta(hours=self.window_size_hours)
        self.error_data = self.error_data[self.error_data['timestamp'] > cutoff_time]
    
    def get_hourly_correction_factor(self, hour):
        """获取特定小时的修正因子"""
        hour_data = self.error_data[self.error_data['hour'] == hour]
        
        if len(hour_data) < 3:
            return 1.0
        
        # 计算加权平均误差比率（最近的数据权重更高）
        weights = np.exp(np.linspace(-1, 0, len(hour_data)))
        weights = weights / weights.sum()
        
        weighted_ratio = np.average(hour_data['error_ratio'].values, weights=weights)
        
        # 限制修正因子范围
        return max(0.7, min(1.3, weighted_ratio))
    
    def get_pattern_statistics(self):
        """获取误差模式统计信息"""
        if len(self.error_data) == 0:
            return {}
        
        stats = {
            'mean_error_ratio': self.error_data['error_ratio'].mean(),
            'std_error_ratio': self.error_data['error_ratio'].std(),
            'hourly_patterns': {}
        }
        
        for hour in range(24):
            hour_data = self.error_data[self.error_data['hour'] == hour]
            if len(hour_data) > 0:
                stats['hourly_patterns'][hour] = {
                    'mean': hour_data['error_ratio'].mean(),
                    'std': hour_data['error_ratio'].std(),
                    'count': len(hour_data)
                }
        
        return stats

def perform_rolling_forecast_with_pid_correction(
    data_path, 
    start_date, 
    end_date, 
    forecast_interval=15,
    peak_hours=(8, 20), 
    valley_hours=(0, 6),
    peak_weight=5.0, 
    valley_weight=1.5,
    apply_smoothing=False,
    dataset_id='上海',
    forecast_type='load',
    historical_days=8,
    # PID相关参数
    pretrain_days=3,
    window_size_hours=72,
    pid_params=None,
    enable_adaptation=True,
    **kwargs
):
    """
    使用PID控制器进行滚动预测修正
    
    参数:
    - pretrain_days: 预训练天数（默认3天）
    - window_size_hours: 滑动窗口大小（默认72小时）
    - pid_params: PID参数字典 {'kp': 0.8, 'ki': 0.1, 'kd': 0.2}
    - enable_adaptation: 是否启用PID参数自适应
    """
    print(f"\n=== 使用PID误差修正的滚动预测 ===")
    print(f"预测期间: {start_date} 到 {end_date}")
    print(f"时间间隔: {forecast_interval}分钟")
    print(f"数据集ID: {dataset_id}")
    print(f"预训练天数: {pretrain_days}")
    print(f"滑动窗口: {window_size_hours}小时")
    print(f"自适应PID: {'启用' if enable_adaptation else '禁用'}")
    
    # 默认PID参数
    if pid_params is None:
        pid_params = {
            'peak': {'kp': 0.8, 'ki': 0.1, 'kd': 0.2},
            'valley': {'kp': 0.6, 'ki': 0.05, 'kd': 0.1},
            'normal': {'kp': 0.7, 'ki': 0.08, 'kd': 0.15}
        }
    
    # 初始化组件
    error_analyzer = SlidingWindowErrorAnalyzer(window_size_hours)
    pid_controllers = {}
    
    # 为每个小时创建PID控制器
    for hour in range(24):
        if peak_hours[0] <= hour <= peak_hours[1]:
            params = pid_params.get('peak', pid_params)
        elif valley_hours[0] <= hour <= valley_hours[1]:
            params = pid_params.get('valley', pid_params)
        else:
            params = pid_params.get('normal', pid_params)
        
        pid_controllers[hour] = AdaptivePIDController(
            kp=params.get('kp', 0.7),
            ki=params.get('ki', 0.08),
            kd=params.get('kd', 0.15),
            adaptation_rate=0.01 if enable_adaptation else 0
        )
    
    # 步骤1: 预训练阶段
    start_datetime = pd.to_datetime(start_date)
    pretrain_start = start_datetime - timedelta(days=pretrain_days)
    pretrain_end = start_datetime - timedelta(minutes=forecast_interval)
    
    print(f"\n开始预训练阶段: {pretrain_start} 到 {pretrain_end}")
    
    try:
        pretrain_results = perform_rolling_forecast_with_peak_awareness(
            data_path=data_path,
            start_date=pretrain_start.strftime('%Y-%m-%d %H:%M:%S'),
            end_date=pretrain_end.strftime('%Y-%m-%d %H:%M:%S'),
            forecast_interval=forecast_interval,
            peak_hours=peak_hours,
            valley_hours=valley_hours,
            peak_weight=peak_weight,
            valley_weight=valley_weight,
            apply_smoothing=False,
            dataset_id=dataset_id,
            forecast_type=forecast_type,
            historical_days=historical_days
        )
        
        # 从预训练结果更新误差分析器
        for _, row in pretrain_results.iterrows():
            if not pd.isna(row['actual']) and row['predicted'] > 0:
                timestamp = pd.to_datetime(row['datetime'])
                error_analyzer.update(
                    timestamp=timestamp,
                    actual=row['actual'],
                    predicted=row['predicted'],
                    hour=timestamp.hour
                )
                
                # 让PID控制器学习初始误差
                error = (row['actual'] / row['predicted']) - 1.0
                pid_controllers[timestamp.hour].compute(error)
        
        print(f"预训练完成，收集了 {len(error_analyzer.error_data)} 个误差样本")
        
    except Exception as e:
        print(f"预训练阶段出错: {e}，将使用默认参数继续")
    
    # 步骤2: 正式预测阶段
    print(f"\n开始正式预测阶段: {start_date} 到 {end_date}")
    
    # 执行基础预测
    base_results = perform_rolling_forecast_with_peak_awareness(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        forecast_interval=forecast_interval,
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=peak_weight,
        valley_weight=valley_weight,
        apply_smoothing=False,
        dataset_id=dataset_id,
        forecast_type=forecast_type,
        historical_days=historical_days
    )
    
    # 应用PID修正
    corrected_results = []
    
    for idx, row in base_results.iterrows():
        timestamp = pd.to_datetime(row['datetime'])
        hour = timestamp.hour
        predicted_raw = row['predicted']
        actual_value = row.get('actual', np.nan)
        
        # 获取滑动窗口修正因子
        window_factor = error_analyzer.get_hourly_correction_factor(hour)
        
        # 获取PID修正因子
        if not pd.isna(actual_value) and predicted_raw > 0:
            # 如果有实际值，更新误差分析器
            error_analyzer.update(timestamp, actual_value, predicted_raw, hour)
            error = (actual_value / predicted_raw) - 1.0
        else:
            # 使用历史平均误差
            error = window_factor - 1.0
        
        pid_factor = pid_controllers[hour].compute(error)
        
        # 组合修正因子（加权平均）
        combined_factor = 0.7 * window_factor + 0.3 * pid_factor
        
        # 应用修正
        predicted_corrected = predicted_raw * combined_factor
        
        # 记录结果
        corrected_results.append({
            'datetime': timestamp,
            'actual': actual_value,
            'predicted_raw': predicted_raw,
            'predicted': predicted_corrected,
            'window_factor': window_factor,
            'pid_factor': pid_factor,
            'combined_factor': combined_factor,
            'is_peak': row.get('is_peak', False)
        })
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(corrected_results)
    
    # 应用平滑处理（如果需要）
    if apply_smoothing and len(results_df) > 0:
        from scripts.forecast.forecast import apply_adaptive_smoothing
        smoothed_predictions = apply_adaptive_smoothing(
            results_df['predicted'].values,
            results_df['datetime'].values
        )
        results_df['predicted_smoothed'] = smoothed_predictions
    else:
        results_df['predicted_smoothed'] = results_df['predicted']
    
    # 获取误差统计信息
    error_stats = error_analyzer.get_pattern_statistics()
    
    # 保存详细结果和统计信息
    results_dir = f"results/pid_correction/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存预测结果
    csv_path = f"{results_dir}/pid_forecast_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    
    # 保存统计信息
    stats_path = f"{results_dir}/pid_stats_{timestamp}.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'error_statistics': error_stats,
            'pid_parameters': {
                hour: {
                    'kp': pid_controllers[hour].kp,
                    'ki': pid_controllers[hour].ki,
                    'kd': pid_controllers[hour].kd
                } for hour in range(24)
            },
            'configuration': {
                'pretrain_days': pretrain_days,
                'window_size_hours': window_size_hours,
                'enable_adaptation': enable_adaptation
            }
        }, f, indent=4)
    
    print(f"\n预测完成，结果已保存至: {csv_path}")
    
    # 计算并显示性能指标
    valid_results = results_df.dropna(subset=['actual', 'predicted'])
    if len(valid_results) > 0:
        metrics = calculate_metrics(valid_results['actual'], valid_results['predicted'])
        metrics_raw = calculate_metrics(valid_results['actual'], valid_results['predicted_raw'])
        
        print("\n性能对比:")
        print(f"原始预测 - MAE: {metrics_raw['mae']:.2f}, MAPE: {metrics_raw['mape']:.2f}%")
        print(f"PID修正 - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2f}%")
        print(f"改进率: {((metrics_raw['mae'] - metrics['mae']) / metrics_raw['mae'] * 100):.1f}%")
    
    return results_df

def advanced_error_correction_system(
    data_path, 
    start_date, 
    end_date,
    forecast_interval=15,
    dataset_id='上海',
    forecast_type='load',
    peak_hours=(8, 20),
    valley_hours=(0, 6),
    peak_weight=5.0,
    valley_weight=1.5,
    apply_smoothing=False,
    historical_days=8,
    # 高级参数
    pretrain_days=3,
    window_size_hours=72,
    enable_pid=True,
    enable_pattern_learning=True,
    enable_ensemble=False,
    **kwargs
):
    """
    高级误差修正系统 - 主入口函数
    
    支持多种修正策略：
    1. PID控制修正
    2. 模式学习修正
    3. 集成修正
    """
    print(f"\n{'='*60}")
    print(f"高级误差修正系统")
    print(f"{'='*60}")
    print(f"预测类型: {forecast_type}")
    print(f"省份: {dataset_id}")
    print(f"预测期间: {start_date} 至 {end_date}")
    print(f"启用策略: PID={enable_pid}, 模式学习={enable_pattern_learning}, 集成={enable_ensemble}")
    
    results = []
    
    # 策略1: PID修正
    if enable_pid:
        print("\n执行PID误差修正...")
        pid_results = perform_rolling_forecast_with_pid_correction(
            data_path=data_path,
            start_date=start_date,
            end_date=end_date,
            forecast_interval=forecast_interval,
            peak_hours=peak_hours,
            valley_hours=valley_hours,
            peak_weight=peak_weight,
            valley_weight=valley_weight,
            apply_smoothing=apply_smoothing,
            dataset_id=dataset_id,
            forecast_type=forecast_type,
            historical_days=historical_days,
            pretrain_days=pretrain_days,
            window_size_hours=window_size_hours,
            **kwargs
        )
        results.append(('pid', pid_results))
    
    # 策略2: 模式学习修正（可选）
    if enable_pattern_learning:
        print("\n执行模式学习修正...")
        # 这里可以添加基于机器学习的误差模式识别和修正
        # pattern_results = perform_pattern_based_correction(...)
        # results.append(('pattern', pattern_results))
        pass
    
    # 策略3: 集成修正（可选）
    if enable_ensemble and len(results) > 1:
        print("\n执行集成修正...")
        # 组合多种修正策略的结果
        # ensemble_results = perform_ensemble_correction(results)
        # return ensemble_results
        pass
    
    # 返回主要结果
    if results:
        return results[0][1]  # 返回第一个策略的结果
    else:
        # 如果没有启用任何策略，返回基础预测
        return perform_rolling_forecast_with_peak_awareness(
            data_path=data_path,
            start_date=start_date,
            end_date=end_date,
            forecast_interval=forecast_interval,
            peak_hours=peak_hours,
            valley_hours=valley_hours,
            peak_weight=peak_weight,
            valley_weight=valley_weight,
            apply_smoothing=apply_smoothing,
            dataset_id=dataset_id,
            forecast_type=forecast_type,
            historical_days=historical_days
        )