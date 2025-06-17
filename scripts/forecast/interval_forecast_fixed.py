#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化版区间预测模型
基于已训练的峰值感知模型，使用误差分布生成预测区间
主要优化：
1. 批量预测减少模型加载次数
2. 向量化误差计算
3. 缓存中间结果
4. 并行处理历史误差计算
5. 优化数据预处理流程
6. 修正模型和缩放器路径处理问题
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import pickle
import hashlib
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 导入项目相关模块
from utils.scaler_manager import ScalerManager
from models.torch_models import PeakAwareConvTransformer
from data.dataset_builder import DatasetBuilder

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("interval_forecast_optimized")

class OptimizedIntervalPredictor:
    """优化的区间预测器，支持批量处理和缓存"""
    
    def __init__(self, model_dir, scaler_dir, seq_length=96, device='cpu'):
        self.model_dir = model_dir
        self.scaler_dir = scaler_dir
        self.seq_length = seq_length
        self.device = torch.device(device)
        
        # 延迟加载的组件
        self.model = None
        self.scaler_manager = None
        self.dataset_builder = None
        
        # 立即初始化 dataset_builder，避免 None 错误
        self._initialize_dataset_builder()
        
        # 缓存和统计
        self._feature_cache = {}
        self._error_cache = {}
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_predictions': 0,
            'batch_predictions': 0
        }
    
    def _initialize_dataset_builder(self):
        """立即初始化 dataset_builder"""
        try:
            from data.dataset_builder import DatasetBuilder
            self.dataset_builder = DatasetBuilder(seq_length=self.seq_length, pred_horizon=1)
            logger.info("DatasetBuilder 初始化成功")
        except Exception as e:
            logger.error(f"DatasetBuilder 初始化失败: {e}")
            # 创建一个简单的替代品避免 None 错误
            self.dataset_builder = self._create_fallback_dataset_builder()
    
    def _create_fallback_dataset_builder(self):
        """创建一个简单的替代 DatasetBuilder"""
        class FallbackDatasetBuilder:
            def __init__(self, seq_length, pred_horizon):
                self.seq_length = seq_length
                self.pred_horizon = pred_horizon
            
            def build_dataset_with_peak_awareness(self, df, date_column, value_column, 
                                                interval, peak_hours, valley_hours, 
                                                start_date, end_date, **kwargs):
                logger.warning("使用简化的 DatasetBuilder")
                # 简单返回原始数据框的副本
                result = df.copy()
                # 添加一些基本特征
                if date_column is None and isinstance(df.index, pd.DatetimeIndex):
                    result['hour'] = df.index.hour
                    result['is_peak'] = ((df.index.hour >= peak_hours[0]) & 
                                       (df.index.hour <= peak_hours[1])).astype(int)
                return result
        
        return FallbackDatasetBuilder(self.seq_length, 1)
    
    def _load_components(self):
        """延迟加载模型和缩放器"""
        if self.model is None:
            self._load_model()
        if self.scaler_manager is None:
            self._load_scaler()
        # dataset_builder 已在 __init__ 中初始化
        if self.dataset_builder is None:
            self._initialize_dataset_builder()

    
    def _load_model(self):
        """加载模型 - 按照forecast.py的方式"""
        try:
            # 检查模型目录是否存在
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
            
            # 检查目录中是否有.pth文件
            if not any(f.endswith('.pth') for f in os.listdir(self.model_dir)):
                raise FileNotFoundError(f"模型目录中没有.pth文件: {self.model_dir}")
            
            # 使用目录路径加载模型（与forecast.py保持一致）
            self.model = PeakAwareConvTransformer.load(save_dir=self.model_dir)
            
            if hasattr(self.model, 'forecaster') and self.model.forecaster:
                if hasattr(self.model.forecaster, 'model'):
                    self.model.forecaster.model = self.model.forecaster.model.to(self.device)
                    self.model.forecaster.device = self.device
                    self.model.forecaster.model.eval()
            
            logger.info(f"模型加载成功，从目录: {self.model_dir}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _load_scaler(self):
        """加载缩放器 - 按照forecast.py的方式"""
        try:
            # 检查缩放器目录是否存在
            if not os.path.exists(self.scaler_dir):
                raise FileNotFoundError(f"缩放器目录不存在: {self.scaler_dir}")
            
            # 使用目录路径初始化ScalerManager（与forecast.py保持一致）
            self.scaler_manager = ScalerManager(scaler_path=self.scaler_dir)
            
            # 验证缩放器是否正确加载
            x_scaler = self.scaler_manager.load_scaler('X')
            y_scaler = self.scaler_manager.load_scaler('y')
            
            if x_scaler is not None and y_scaler is not None:
                logger.info(f"缩放器加载成功，从目录: {self.scaler_dir}")
            else:
                logger.warning("缩放器加载不完整")
                
        except Exception as e:
            logger.warning(f"缩放器加载失败: {e}")
            self.scaler_manager = None
    
    def _generate_cache_key(self, start_time, end_time, data_hash):
        """生成缓存键"""
        return f"{data_hash}_{start_time}_{end_time}_{self.seq_length}"
    
    def _hash_data(self, data):
        """计算数据哈希值"""
        return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()[:16]
    
    def batch_prepare_features(self, historical_data, time_points, forecast_type, 
                             peak_hours, valley_hours, fix_nan=True, batch_size=50):
        """批量准备特征数据"""
        features_batch = []
        data_hash = self._hash_data(historical_data)
        
        logger.info(f"批量准备 {len(time_points)} 个时间点的特征数据")
        
        for i, time_point in enumerate(time_points):
            if i % 20 == 0:
                logger.debug(f"准备特征进度: {i+1}/{len(time_points)}")
                
            cache_key = self._generate_cache_key(time_point, time_point, data_hash)
            
            # 检查缓存
            if cache_key in self._feature_cache:
                features_batch.append(self._feature_cache[cache_key])
                self.stats['cache_hits'] += 1
                continue
            
            self.stats['cache_misses'] += 1
            
            try:
                # 准备特征数据的时间范围
                feature_end = time_point - timedelta(minutes=15)
                feature_start = feature_end - timedelta(days=14)
                
                # 提取特征数据
                feature_data = historical_data.loc[feature_start:feature_end].copy()
                
                if len(feature_data) < 24*4:  # 至少需要1天数据
                    fallback_value = historical_data[forecast_type].mean()
                    feature_item = {
                        'features': None, 
                        'fallback_value': fallback_value,
                        'valid': False
                    }
                    features_batch.append(feature_item)
                    self._feature_cache[cache_key] = feature_item
                    continue
                
                # 修复NaN值
                if fix_nan and feature_data.isna().sum().sum() > 0:
                    feature_data = feature_data.ffill().bfill().fillna(0)
                
                # 构建增强数据集
                enhanced_data = self.dataset_builder.build_dataset_with_peak_awareness(
                    df=feature_data,
                    date_column=None,
                    value_column=forecast_type,
                    interval=15,
                    peak_hours=peak_hours,
                    valley_hours=valley_hours,
                    start_date=feature_start,
                    end_date=feature_end
                )
                
                if fix_nan and enhanced_data.isna().sum().sum() > 0:
                    enhanced_data = enhanced_data.fillna(0)
                
                if len(enhanced_data) < self.seq_length:
                    fallback_value = historical_data[forecast_type].mean()
                    feature_item = {
                        'features': None, 
                        'fallback_value': fallback_value,
                        'valid': False
                    }
                    features_batch.append(feature_item)
                    self._feature_cache[cache_key] = feature_item
                    continue
                
                # 提取特征
                X = enhanced_data.iloc[-self.seq_length:].drop(columns=[forecast_type]).values
                
                if fix_nan and np.isnan(X).any():
                    X = np.nan_to_num(X, nan=0.0)
                
                # 计算备用值
                current_hour = time_point.hour
                same_hour_data = [v for j, v in enumerate(historical_data[forecast_type]) 
                                if historical_data.index[j].hour == current_hour]
                fallback_value = np.mean(same_hour_data) if same_hour_data else historical_data[forecast_type].mean()
                
                feature_item = {
                    'features': X,
                    'fallback_value': fallback_value,
                    'valid': True
                }
                
                features_batch.append(feature_item)
                self._feature_cache[cache_key] = feature_item
                
            except Exception as e:
                logger.error(f"准备特征数据失败 {time_point}: {e}")
                fallback_value = historical_data[forecast_type].mean()
                feature_item = {
                    'features': None, 
                    'fallback_value': fallback_value,
                    'valid': False
                }
                features_batch.append(feature_item)
                self._feature_cache[cache_key] = feature_item
        
        valid_count = sum(1 for item in features_batch if item['valid'])
        logger.info(f"特征准备完成: {valid_count}/{len(features_batch)} 个有效特征")
        
        return features_batch
    
    def batch_predict(self, features_batch, batch_size=32):
        """批量预测"""
        self._load_components()
        
        # 分离有效和无效的特征
        valid_indices = [i for i, item in enumerate(features_batch) if item['valid']]
        invalid_indices = [i for i, item in enumerate(features_batch) if not item['valid']]
        
        predictions = [None] * len(features_batch)
        
        # 对无效特征使用备用值
        for i in invalid_indices:
            predictions[i] = features_batch[i]['fallback_value']
        
        # 批量处理有效特征
        if valid_indices:
            logger.info(f"开始批量预测 {len(valid_indices)} 个有效样本")
            
            # 准备批量数据
            batch_features = np.stack([features_batch[i]['features'] for i in valid_indices])
            
            # 标准化
            if self.scaler_manager:
                original_shape = batch_features.shape
                batch_features_scaled = self.scaler_manager.transform(
                    'X', batch_features.reshape(len(batch_features), -1)
                ).reshape(original_shape)
            else:
                batch_features_scaled = batch_features
            
            # 批量预测
            with torch.no_grad():
                X_tensor = torch.FloatTensor(batch_features_scaled).to(self.device)
                
                if torch.isnan(X_tensor).any():
                    X_tensor = torch.nan_to_num(X_tensor)
                
                # 分批预测以避免内存问题
                batch_predictions = []
                
                for start_idx in range(0, len(X_tensor), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_tensor))
                    batch_X = X_tensor[start_idx:end_idx]
                    
                    self.stats['batch_predictions'] += 1
                    
                    if hasattr(self.model, 'predict'):
                        # 逐个预测（如果模型不支持真正的批量预测）
                        for j in range(len(batch_X)):
                            try:
                                raw_pred = self.model.predict(batch_X[j:j+1].cpu().numpy())
                                if raw_pred.ndim == 1:
                                    raw_pred = raw_pred.reshape(1, -1)
                                
                                # 检查预测值合理性
                                if np.abs(raw_pred).max() > 100:
                                    raw_pred = np.clip(raw_pred, -10, 10)
                                
                                # 反标准化
                                if self.scaler_manager:
                                    pred_inverse = self.scaler_manager.inverse_transform('y', raw_pred)
                                    predicted_value = pred_inverse.flatten()[0]
                                else:
                                    predicted_value = raw_pred.flatten()[0]
                                
                                # 检查预测值合理性
                                if (np.isnan(predicted_value) or np.isinf(predicted_value) 
                                    or np.abs(predicted_value) > 1e6):
                                    predicted_value = features_batch[valid_indices[start_idx + j]]['fallback_value']
                                
                                batch_predictions.append(max(0, predicted_value))
                                self.stats['total_predictions'] += 1
                                
                            except Exception as pred_err:
                                logger.error(f"单个预测失败: {pred_err}")
                                batch_predictions.append(
                                    features_batch[valid_indices[start_idx + j]]['fallback_value']
                                )
                
                # 将预测结果分配到正确位置
                for i, valid_idx in enumerate(valid_indices):
                    if i < len(batch_predictions):
                        predictions[valid_idx] = batch_predictions[i]
                    else:
                        predictions[valid_idx] = features_batch[valid_idx]['fallback_value']
        
        logger.info(f"批量预测完成，共 {len(predictions)} 个预测值")
        return predictions
    
    def get_stats(self):
        """获取性能统计信息"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'total_requests': total_requests,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'total_predictions': self.stats['total_predictions'],
            'batch_predictions': self.stats['batch_predictions']
        }

def calculate_optimal_beta_vectorized(errors, alpha=0.1):
    """向量化计算最优beta参数"""
    sorted_errors = np.sort(errors)
    n = len(sorted_errors)
    
    if n < 10:  # 如果样本太少，使用默认beta
        return alpha / 2
    
    # 创建beta值数组进行向量化计算
    max_beta_idx = min(int(n * alpha), n - 1)
    if max_beta_idx <= 0:
        return alpha / 2
        
    beta_indices = np.arange(0, max_beta_idx + 1)
    betas = beta_indices / n
    
    # 向量化计算所有beta对应的区间宽度
    lower_indices = beta_indices
    upper_indices = np.minimum(beta_indices + int(n * (1 - alpha)), n - 1)
    
    # 计算所有区间宽度
    widths = sorted_errors[upper_indices] - sorted_errors[lower_indices]
    
    # 找到最小宽度对应的beta
    min_width_idx = np.argmin(widths)
    optimal_beta = betas[min_width_idx]
    
    return optimal_beta

def generate_prediction_intervals_optimized(point_predictions, historical_errors, confidence_level=0.9):
    """优化的预测区间生成"""
    point_predictions = np.array(point_predictions).flatten()
    
    # 处理历史误差
    if isinstance(historical_errors, list):
        try:
            historical_errors = np.array(historical_errors).flatten()
        except Exception as e:
            logger.warning(f"无法将历史误差转换为数组: {e}")
            historical_errors = np.random.normal(0, np.std(point_predictions) * 0.1, 1000)
    elif isinstance(historical_errors, np.ndarray):
        historical_errors = historical_errors.flatten()
    
    # 移除无效值
    historical_errors = historical_errors[~(np.isnan(historical_errors) | np.isinf(historical_errors))]
    
    if len(historical_errors) == 0:
        logger.warning("没有有效的历史误差，使用默认误差分布")
        historical_errors = np.random.normal(0, np.std(point_predictions) * 0.1, 1000)
    
    alpha = 1 - confidence_level
    
    # 使用向量化的beta计算
    beta = calculate_optimal_beta_vectorized(historical_errors, alpha)
    
    # 计算误差的分位数
    sorted_errors = np.sort(historical_errors)
    n = len(sorted_errors)
    
    lower_idx = max(0, min(int(n * beta), n-1))
    upper_idx = max(0, min(int(n * (1 - alpha + beta)), n-1))
    
    lower_error = sorted_errors[lower_idx]
    upper_error = sorted_errors[upper_idx]
    
    # 生成预测区间
    lower_bounds = point_predictions + lower_error
    upper_bounds = point_predictions + upper_error

    # 保证 lower_bound <= predicted <= upper_bound
    lower_bounds = np.minimum(lower_bounds, point_predictions)
    upper_bounds = np.maximum(upper_bounds, point_predictions)

    # 如果 lower_bound > upper_bound，强制交换
    swap_mask = lower_bounds > upper_bounds
    if np.any(swap_mask):
        temp = lower_bounds[swap_mask].copy()
        lower_bounds[swap_mask] = upper_bounds[swap_mask]
        upper_bounds[swap_mask] = temp

    # 保证下界不为负
    lower_bounds = np.maximum(0, lower_bounds)
    
    logger.info(f"区间参数: beta={beta:.4f}, lower_error={lower_error:.2f}, upper_error={upper_error:.2f}")
    
    return lower_bounds, upper_bounds

def perform_interval_forecast(data_path, forecast_type, province, time_point, n_intervals, 
                            model_path, scaler_path, device='cpu', quantiles=None, rolling=False, 
                            seq_length=96, peak_hours=(9, 20), valley_hours=(0, 6), fix_nan=True):
    """
    优化版区间预测函数 - 修正模型和缩放器路径处理
    
    Args:
        data_path: 时间序列数据路径
        forecast_type: 预测类型（load/pv/wind）
        province: 省份名称
        time_point: 预测时间点
        n_intervals: 历史间隔数量
        model_path: 模型路径（可以是文件路径或目录路径）
        scaler_path: 缩放器路径（可以是文件路径或目录路径）
        device: 计算设备
        quantiles: 分位数（用于计算置信水平）
        rolling: 是否使用滚动预测
        seq_length: 序列长度
        peak_hours: 高峰时段
        valley_hours: 低谷时段
        fix_nan: 是否修复NaN值
        
    Returns:
        预测结果DataFrame和指标字典
    """
    logger.info(f"===== 开始优化版区间预测: {forecast_type}, {time_point} =====")
    
    start_time = datetime.now()
    
    try:
        # 处理模型和缩放器路径 - 兼容文件路径和目录路径
        if os.path.isfile(model_path):
            model_dir = os.path.dirname(model_path)
            logger.info(f"检测到模型文件路径，提取目录: {model_dir}")
        else:
            model_dir = model_path
            logger.info(f"使用模型目录路径: {model_dir}")
        
        if os.path.isfile(scaler_path):
            scaler_dir = os.path.dirname(scaler_path)
            logger.info(f"检测到缩放器文件路径，提取目录: {scaler_dir}")
        else:
            scaler_dir = scaler_path
            logger.info(f"使用缩放器目录路径: {scaler_dir}")
        
        # 初始化优化预测器
        predictor = OptimizedIntervalPredictor(model_dir, scaler_dir, seq_length, device)
        
        # 转换时间点
        if isinstance(time_point, str):
            time_point = pd.to_datetime(time_point)
        
        # 设置置信水平
        if quantiles:
            confidence_level = quantiles[1] - quantiles[0]
        else:
            confidence_level = 0.9
        
        logger.info(f"置信水平: {confidence_level}")
        
        # 加载时间序列数据
        ts_data = pd.read_csv(data_path, index_col=0)
        ts_data.index = pd.to_datetime(ts_data.index)
        logger.info(f"加载数据: {len(ts_data)} 行，时间范围 {ts_data.index.min()} 至 {ts_data.index.max()}")
        
        # 如果当前数据文件没有天气数据，但需要进行天气场景分析，尝试加载包含天气数据的文件
        has_weather_data = any('weather' in col for col in ts_data.columns)
        weather_data = None
        
        if forecast_type == 'load' and not has_weather_data:
            # 尝试加载对应的天气数据文件
            weather_data_path = data_path.replace(f'timeseries_{forecast_type}_', f'timeseries_{forecast_type}_weather_')
            if os.path.exists(weather_data_path):
                try:
                    weather_data = pd.read_csv(weather_data_path, index_col=0)
                    weather_data.index = pd.to_datetime(weather_data.index)
                    logger.info(f"成功加载天气数据文件: {weather_data_path}")
                    logger.info(f"天气数据: {len(weather_data)} 行，包含列: {list(weather_data.columns)}")
                    
                    # 检查天气数据是否包含天气列
                    has_weather_in_weather_data = any('weather' in col for col in weather_data.columns)
                    if has_weather_in_weather_data:
                        logger.info("天气数据文件包含天气列，将用于场景分析")
                    else:
                        logger.warning("天气数据文件不包含天气列")
                        weather_data = None
                        
                except Exception as e:
                    logger.warning(f"加载天气数据文件失败: {e}")
                    weather_data = None
            else:
                logger.info(f"天气数据文件不存在: {weather_data_path}")
        
        # 设置时间范围
        history_end = time_point - timedelta(minutes=15)
        history_start = history_end - timedelta(days=n_intervals)
        required_hist_start = history_start - timedelta(days=10)
        
        forecast_start = time_point.replace(hour=0, minute=0, second=0)
        forecast_end = (time_point + timedelta(days=1)).replace(hour=0, minute=0, second=0) - timedelta(minutes=15)
        forecast_times = pd.date_range(start=forecast_start, end=forecast_end, freq='15min')
        
        # 检查是否有预测日期的实际数据
        actual_data_available = any(ts_data.index.date == time_point.date())
        logger.info(f"预测日期 {time_point.date()} 的实际数据可用性: {actual_data_available}")
        
        if actual_data_available:
            actual_day_data = ts_data[ts_data.index.date == time_point.date()].copy()
            logger.info(f"找到预测日实际数据: {len(actual_day_data)} 个数据点")
        else:
            logger.warning(f"未找到预测日期 {time_point.date()} 的实际数据")
            actual_day_data = pd.DataFrame()
        
        # 确保有足够的历史数据
        if ts_data.index.min() > required_hist_start:
            required_hist_start = ts_data.index.min()
            history_start = required_hist_start + timedelta(days=7)
        
        logger.info(f"历史数据范围: {required_hist_start} 至 {history_end}")
        logger.info(f"预测时间点数量: {len(forecast_times)}")
        
        # 获取历史数据
        historical_data = ts_data.loc[required_hist_start:history_end].copy()
        
        # 修复缺失值
        if fix_nan and historical_data.isna().sum().sum() > 0:
            logger.info("修复历史数据缺失值...")
            historical_data = historical_data.ffill().bfill().fillna(0)
        
        # ========== 天气场景分析与典型场景匹配 ==========
        weather_scenario_info = {}
        scenario_match_info = {}
        
        if forecast_type == 'load':  # 只对负荷预测进行天气场景分析
            try:
                from utils.scenario_matcher import ScenarioMatcher
                matcher = ScenarioMatcher(province)
                
                # 检查是否有天气数据可用（来自主数据文件或单独的天气数据文件）
                has_weather_data_main = any('weather' in col for col in historical_data.columns)
                has_weather_data_separate = weather_data is not None and any('weather' in col for col in weather_data.columns)
                
                if has_weather_data_main or has_weather_data_separate:
                    # 有天气数据，进行完整的天气场景分析
                    from utils.weather_scenario_classifier import WeatherScenarioClassifier
                    classifier = WeatherScenarioClassifier()
                    
                    # 选择天气数据源
                    if has_weather_data_main:
                        weather_source = historical_data.copy()
                        logger.info("使用主数据文件中的天气数据进行场景分析")
                    else:
                        # 使用单独的天气数据文件，但需要确保时间范围匹配
                        weather_source = weather_data.loc[historical_data.index.min():historical_data.index.max()].copy()
                        logger.info("使用单独的天气数据文件进行场景分析")
                    
                    # 准备天气数据列名
                    if 'weather_temperature_c' in weather_source.columns:
                        weather_source = weather_source.rename(columns={
                            'weather_temperature_c': 'temperature',
                            'weather_wind_speed': 'wind_speed',
                            'weather_relative_humidity': 'humidity',
                            'weather_precipitation_mm': 'precipitation'
                        })
                    
                    # 分析预测日的天气场景
                    forecast_day_data = weather_source[weather_source.index.date == time_point.date()]
                    if not forecast_day_data.empty:
                        scenario_result = classifier.identify_scenario(forecast_day_data)
                        weather_scenario_info = {
                            'scenario_type': scenario_result['name'],
                            'temperature_mean': forecast_day_data['temperature'].mean(),
                            'temperature_max': forecast_day_data['temperature'].max(),
                            'temperature_min': forecast_day_data['temperature'].min(),
                            'humidity_mean': forecast_day_data['humidity'].mean(),
                            'wind_speed_mean': forecast_day_data['wind_speed'].mean(),
                            'precipitation_sum': forecast_day_data['precipitation'].sum()
                        }
                        logger.info(f"预测日天气场景分析结果: {weather_scenario_info}")
                    else:
                        logger.warning(f"在天气数据中未找到预测日 {time_point.date()} 的数据")
                        weather_scenario_info = {
                            'scenario_type': '数据缺失',
                            'temperature_mean': 25.0,
                            'temperature_max': 30.0,
                            'temperature_min': 20.0,
                            'humidity_mean': 60.0,
                            'wind_speed_mean': 3.0,
                            'precipitation_sum': 0.0
                        }
                else:
                    # 没有天气数据，使用默认值创建场景信息（为了支持场景匹配）
                    logger.info("数据中没有天气信息，将使用默认天气参数进行场景匹配")
                    weather_scenario_info = {
                        'scenario_type': '负荷驱动场景',
                        'temperature_mean': 25.0,  # 默认温度
                        'temperature_max': 30.0,
                        'temperature_min': 20.0,
                        'humidity_mean': 60.0,     # 默认湿度
                        'wind_speed_mean': 3.0,    # 默认风速
                        'precipitation_sum': 0.0   # 默认降水
                    }
                    logger.info(f"使用默认天气参数: {weather_scenario_info}")
                
                # 注意：典型场景匹配将在预测完成后进行，因为需要使用预测值计算负荷特征
                        
            except Exception as e:
                logger.warning(f"天气场景分析失败: {e}")
                weather_scenario_info = {'error': str(e)}
        
        # ========== 优化的历史误差计算 ==========
        logger.info("开始优化的历史误差计算...")
        
        # 智能采样历史预测时间点（降低采样频率以提高效率）
        # 根据数据量动态调整采样频率
        total_hours = (history_end - history_start).total_seconds() / 3600
        if total_hours > 168:  # 超过一周
            freq = '3H'  # 每3小时采样
            max_points = 56  # 一周的3小时点数
        elif total_hours > 72:  # 超过3天
            freq = '2H'  # 每2小时采样
            max_points = 84  # 一周的2小时点数
        else:
            freq = '1H'  # 每小时采样
            max_points = 168  # 一周的小时数
        
        historical_times = pd.date_range(
            start=history_start,
            end=history_end,
            freq=freq
        )[:max_points]
        
        logger.info(f"历史预测时间点数量（优化后）: {len(historical_times)}，采样频率: {freq}")
        
        # 批量准备特征
        logger.info("批量准备历史特征数据...")
        features_batch = predictor.batch_prepare_features(
            historical_data, historical_times, forecast_type, peak_hours, valley_hours, fix_nan
        )
        
        # 批量预测
        logger.info("执行批量历史预测...")
        historical_predictions = predictor.batch_predict(features_batch)
        
        # 向量化计算历史误差
        logger.info("计算历史误差...")
        historical_errors = []
        valid_error_count = 0
        
        for i, hist_time in enumerate(historical_times):
            if hist_time in historical_data.index:
                actual_value = historical_data.loc[hist_time, forecast_type]
                if not pd.isna(actual_value) and historical_predictions[i] is not None:
                    error = actual_value - historical_predictions[i]
                    historical_errors.append(error)
                    valid_error_count += 1
        
        # 检查误差样本数量
        if len(historical_errors) < 10:
            logger.warning(f"历史误差样本不足 ({len(historical_errors)} < 10)，使用默认误差分布")
            std_dev = historical_data[forecast_type].std() * 0.1
            historical_errors = np.random.normal(0, std_dev, 200)
        
        logger.info(f"历史误差计算完成，有效样本: {valid_error_count}/{len(historical_times)}")
        logger.info(f"误差统计: 均值={np.mean(historical_errors):.2f}, 标准差={np.std(historical_errors):.2f}")
        
        # ========== 优化的预测日预测 ==========
        logger.info(f"开始对预测日 {time_point.date()} 进行优化预测...")
        
        # 获取预测日历史数据
        forecast_hist_start = forecast_start - timedelta(days=10)
        forecast_hist_data = ts_data.loc[forecast_hist_start:forecast_end].copy()
        
        # 修复预测日数据缺失值
        if fix_nan and forecast_hist_data.isna().sum().sum() > 0:
            forecast_hist_data = forecast_hist_data.ffill().bfill().fillna(0)
        
        forecast_times = pd.date_range(start=forecast_start, end=forecast_end, freq='15min')
        

        # 批量准备预测日特征
        logger.info("批量准备预测日特征数据...")
        forecast_features_batch = predictor.batch_prepare_features(
            forecast_hist_data, forecast_times, forecast_type, peak_hours, valley_hours, fix_nan
        )
        
        # 批量预测预测日
        logger.info("执行预测日批量预测...")
        point_predictions = predictor.batch_predict(forecast_features_batch)
        
        logger.info(f"预测日预测完成，共 {len(point_predictions)} 个预测点")
        logger.info(f"点预测统计: 最小={np.min(point_predictions):.2f}, "
                   f"最大={np.max(point_predictions):.2f}, 均值={np.mean(point_predictions):.2f}")
        
        # ========== 生成预测区间 ==========
        logger.info("生成预测区间...")
        
        lower_bound, upper_bound = generate_prediction_intervals_optimized(
            point_predictions, 
            historical_errors, 
            confidence_level=confidence_level
        )
        
        avg_interval_width = np.mean(upper_bound - lower_bound)
        logger.info(f"平均区间宽度: {avg_interval_width:.2f}")
        
        # ========== 执行典型场景匹配（在预测完成后） ==========
        if forecast_type == 'load' and weather_scenario_info and 'error' not in weather_scenario_info:
            try:
                from utils.scenario_matcher import ScenarioMatcher
                matcher = ScenarioMatcher(province)
                
                # 计算当日平均负荷（用于场景匹配）
                # 首先尝试使用实际数据
                forecast_day_load_data = historical_data[historical_data.index.date == time_point.date()]
                if not forecast_day_load_data.empty and forecast_type in forecast_day_load_data.columns:
                    daily_load_mean = forecast_day_load_data[forecast_type].mean()
                    daily_load_volatility = forecast_day_load_data[forecast_type].std() / daily_load_mean if daily_load_mean > 0 else 0
                    logger.info(f"使用实际负荷数据计算场景特征: 均值={daily_load_mean:.2f}, 波动率={daily_load_volatility:.3f}")
                else:
                    # 如果没有当日负荷数据，使用预测值的统计
                    daily_load_mean = np.mean(point_predictions)
                    daily_load_volatility = np.std(point_predictions) / daily_load_mean if daily_load_mean > 0 else 0.15
                    logger.info(f"使用预测负荷数据计算场景特征: 均值={daily_load_mean:.2f}, 波动率={daily_load_volatility:.3f}")
                
                # 构建用于场景匹配的特征
                current_features = {
                    'temperature_mean': weather_scenario_info['temperature_mean'],
                    'humidity_mean': weather_scenario_info['humidity_mean'],
                    'wind_speed_mean': weather_scenario_info['wind_speed_mean'],
                    'precipitation_sum': weather_scenario_info['precipitation_sum'],
                    'load_mean': daily_load_mean,
                    'load_volatility': daily_load_volatility
                }
                
                # 执行场景匹配
                logger.info("开始执行典型场景匹配...")
                match_result = matcher.match_scenario(current_features, province)
                
                if match_result:
                    scenario_match_info = {
                        'matched_scenario': match_result['matched_scenario']['name'],
                        'similarity': match_result['matched_scenario']['similarity'],
                        'similarity_percentage': match_result['matched_scenario']['similarity_percentage'],
                        'confidence_level': match_result['confidence_level'],
                        'description': match_result['matched_scenario']['description'],
                        'typical_percentage': match_result['matched_scenario']['typical_percentage'],
                        'distance': match_result['matched_scenario']['distance'],
                        'top_scenarios': [
                            {
                                'name': scenario['name'],
                                'similarity_percentage': scenario['similarity_percentage'],
                                'rank': scenario['rank']
                            }
                            for scenario in match_result['all_scenarios'][:3]
                        ],
                        'feature_contributions': match_result['feature_analysis']
                    }
                    
                    # 将场景匹配信息添加到天气场景信息中
                    weather_scenario_info.update({
                        'scenario_match': scenario_match_info,
                        'daily_load_mean': daily_load_mean,
                        'daily_load_volatility': daily_load_volatility
                    })
                    
                    logger.info(f"场景匹配完成: {scenario_match_info['matched_scenario']} "
                              f"(相似度: {scenario_match_info['similarity_percentage']:.1f}%)")
                else:
                    logger.warning("场景匹配失败")
                    weather_scenario_info['scenario_match_error'] = "场景匹配失败"
                    
            except Exception as e:
                logger.warning(f"典型场景匹配执行失败: {e}")
                weather_scenario_info['scenario_match_error'] = f"场景匹配执行失败: {str(e)}"
        
        # ========== 创建结果DataFrame并加载实际数据 ==========
        results_df = pd.DataFrame({
            'datetime': forecast_times,
            'predicted': point_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
        # 添加实际数据
        if not actual_day_data.empty:
            logger.info("正在添加实际数据...")
            
            # 创建实际值列
            results_df['actual'] = np.nan
            
            # 匹配时间并添加实际值
            matched_count = 0
            for i, forecast_time in enumerate(forecast_times):
                # 寻找最接近的实际数据点
                matching_actual = actual_day_data[
                    (actual_day_data.index.hour == forecast_time.hour) & 
                    (actual_day_data.index.minute == forecast_time.minute)
                ]
                
                if not matching_actual.empty:
                    # 如果有多个匹配点，取第一个
                    actual_value = matching_actual[forecast_type].iloc[0]
                    if not pd.isna(actual_value):
                        results_df.at[i, 'actual'] = actual_value
                        matched_count += 1
            
            logger.info(f"成功匹配 {matched_count}/{len(forecast_times)} 个实际数据点")
            
            # 计算命中率（实际值落在预测区间内的比例）
            valid_actual = results_df['actual'].notna()
            if valid_actual.sum() > 0:
                hits = ((results_df.loc[valid_actual, 'actual'] >= results_df.loc[valid_actual, 'lower_bound']) & 
                       (results_df.loc[valid_actual, 'actual'] <= results_df.loc[valid_actual, 'upper_bound']))
                hit_rate = hits.mean() * 100
                logger.info(f"区间预测命中率: {hit_rate:.2f}%")
                
                # 计算预测准确性指标
                valid_data = results_df.dropna(subset=['actual', 'predicted'])
                if not valid_data.empty:
                    mae = np.mean(np.abs(valid_data['actual'] - valid_data['predicted']))
                    mape = np.mean(np.abs((valid_data['actual'] - valid_data['predicted']) / valid_data['actual'])) * 100
                    rmse = np.sqrt(np.mean((valid_data['actual'] - valid_data['predicted']) ** 2))
                    
                    logger.info(f"点预测准确性指标:")
                    logger.info(f"  MAE: {mae:.2f}")
                    logger.info(f"  MAPE: {mape:.2f}%")
                    logger.info(f"  RMSE: {rmse:.2f}")
        else:
            logger.warning("没有可用的实际数据，无法计算命中率")
        
        # 设置datetime为索引
        results_df.set_index('datetime', inplace=True)
        
        # 检查并修复NaN值
        if results_df.isna().any().any():
            logger.warning("最终结果中有NaN值，将进行填充")
            # 只对数值列进行填充，保持actual列的NaN
            numeric_columns = ['predicted', 'lower_bound', 'upper_bound']
            for col in numeric_columns:
                if col in results_df.columns:
                    results_df[col] = results_df[col].ffill().bfill()
        
        # 重置索引并格式化datetime
        results_df = results_df.reset_index()
        if pd.api.types.is_datetime64_dtype(results_df['datetime']):
            results_df['datetime'] = results_df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # 添加时间字符串列
        results_df['time'] = pd.to_datetime(results_df['datetime']).dt.strftime('%H:%M')
        
        # 计算执行时间
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 输出性能统计
        stats = predictor.get_stats()
        logger.info(f"区间预测完成: {len(results_df)} 行，执行时间: {execution_time:.2f}秒")
        logger.info(f"性能统计: {stats}")
        
        # 返回结果DataFrame和指标
        metrics = {
            'avg_interval_width': avg_interval_width,
            'hit_rate': hit_rate if 'hit_rate' in locals() else 0.0,
            'mae': mae if 'mae' in locals() else 0.0,
            'mape': mape if 'mape' in locals() else 0.0,
            'rmse': rmse if 'rmse' in locals() else 0.0,
            'weather_scenario': weather_scenario_info  # 添加天气场景信息
        }
        return results_df, metrics
        
    except Exception as e:
        logger.error(f"区间预测过程中发生错误: {str(e)}\n{traceback.format_exc()}")
        raise ValueError(f"区间预测失败: {str(e)}")
        
    finally:
        logger.info("===== 区间预测过程结束 =====")


def perform_interval_forecast_for_range(data_path, forecast_type, province, start_date_str, end_date_str, n_intervals,
                                        model_path, scaler_path, device='cpu', quantiles=None, rolling=False,
                                        seq_length=96, peak_hours=(9, 20), valley_hours=(0, 6), fix_nan=True):
    """
    为指定日期范围执行区间预测。

    Args:
        data_path: 时间序列数据路径
        forecast_type: 预测类型（load/pv/wind）
        province: 省份名称
        start_date_str: 预测开始日期字符串 (YYYY-MM-DD)
        end_date_str: 预测结束日期字符串 (YYYY-MM-DD)
        n_intervals: 历史间隔数量
        model_path: 模型路径
        scaler_path: 缩放器路径
        device: 计算设备
        quantiles: 分位数
        rolling: 是否使用滚动预测
        seq_length: 序列长度
        peak_hours: 高峰时段
        valley_hours: 低谷时段
        fix_nan: 是否修复NaN值

    Returns:
        包含合并预测结果的DataFrame和合并的指标字典
    """
    logger.info(f"===== 开始范围区间预测: {forecast_type}, {province}, {start_date_str} to {end_date_str} =====")
    
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
    except ValueError as e:
        logger.error(f"无效的日期格式: {start_date_str} or {end_date_str}. 错误: {e}")
        raise ValueError(f"无效的日期格式: {e}")

    if start_date > end_date:
        logger.error(f"开始日期 {start_date_str} 不能晚于结束日期 {end_date_str}")
        raise ValueError("开始日期不能晚于结束日期")

    all_results_dfs = []
    all_metrics_list = []
    
    current_date = start_date
    while current_date <= end_date:
        logger.info(f"正在为日期 {current_date.strftime('%Y-%m-%d')} 执行区间预测...")
        try:
            daily_results_df, daily_metrics = perform_interval_forecast(
                data_path=data_path,
                forecast_type=forecast_type,
                province=province,
                time_point=current_date,  # 传递datetime对象
                n_intervals=n_intervals,
                model_path=model_path,
                scaler_path=scaler_path,
                device=device,
                quantiles=quantiles,
                rolling=rolling,
                seq_length=seq_length,
                peak_hours=peak_hours,
                valley_hours=valley_hours,
                fix_nan=fix_nan
            )
            if daily_results_df is not None and not daily_results_df.empty:
                all_results_dfs.append(daily_results_df)
                if daily_metrics: # 确保 daily_metrics 不是 None
                    all_metrics_list.append(daily_metrics)
            else:
                logger.warning(f"日期 {current_date.strftime('%Y-%m-%d')} 的预测结果为空，跳过。")
        except Exception as e:
            logger.error(f"日期 {current_date.strftime('%Y-%m-%d')} 的区间预测失败: {e}")
            #可以选择跳过失败的日期或者直接抛出异常
            #这里选择记录错误并继续
        current_date += timedelta(days=1)

    if not all_results_dfs:
        logger.warning("没有成功的预测结果可以合并。")
        return pd.DataFrame(), {} # 返回空的DataFrame和空的metrics字典

    # 合并所有DataFrame
    final_results_df = pd.concat(all_results_dfs, ignore_index=True)
    
    # 合并指标 (数值型指标取平均，非数值型指标取第一个有效值)
    final_metrics = {}
    if all_metrics_list:
        # 初始化一个字典来存储所有指标的总和和计数
        summed_metrics = {}
        count_metrics = {}
        special_metrics = {}  # 存储非数值型指标
        
        for metrics_dict in all_metrics_list:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)) and not pd.isna(value): # 确保是数字且非NaN
                    summed_metrics[key] = summed_metrics.get(key, 0) + value
                    count_metrics[key] = count_metrics.get(key, 0) + 1
                elif key not in special_metrics and value is not None:
                    # 对于非数值型指标（如weather_scenario），取第一个有效值
                    special_metrics[key] = value
        
        # 计算数值型指标的平均值
        for key in summed_metrics:
            if count_metrics[key] > 0:
                final_metrics[key] = summed_metrics[key] / count_metrics[key]
            else:
                final_metrics[key] = None
        
        # 添加非数值型指标
        final_metrics.update(special_metrics)

    logger.info(f"===== 范围区间预测完成 =====")
    return final_results_df, final_metrics

def run_interval_forecast():
    """命令行运行区间预测主函数"""
    parser = argparse.ArgumentParser(description='优化版区间预测脚本')
    parser.add_argument('--forecast_type', choices=['load', 'pv', 'wind'], default='load',
                       help='预测类型：负荷(load)、光伏(pv)或风电(wind)')
    parser.add_argument('--province', required=True, help='省份名称')
    parser.add_argument('--forecast_date', required=True, help='预测日期，格式为YYYY-MM-DD')
    parser.add_argument('--confidence_level', type=float, default=0.9,
                        help='置信水平，默认0.9 (90%)')
    parser.add_argument('--historical_days', type=int, default=14,
                        help='用于计算历史误差的天数，默认14')
    parser.add_argument('--interval', type=int, default=15,
                        help='时间间隔（分钟），默认15')
    parser.add_argument('--save', action='store_true', help='是否保存预测结果')
    parser.add_argument('--no_fix_nan', action='store_true', help='不自动修复NaN值')
    parser.add_argument('--device', type=str, default='auto', help='计算设备: cpu, cuda, auto')
    
    args = parser.parse_args()
    
    print("优化版区间预测脚本启动，参数:")
    print(f"  预测类型: {args.forecast_type}")
    print(f"  省份: {args.province}")
    print(f"  预测日期: {args.forecast_date}")
    print(f"  置信水平: {args.confidence_level}")
    print(f"  历史天数: {args.historical_days}")
    print(f"  时间间隔: {args.interval}分钟")
    print(f"  自动修复NaN: {not args.no_fix_nan}")
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"  计算设备: {device}")
    
    # 执行区间预测
    try:
        data_path = f"data/timeseries_{args.forecast_type}_{args.province}.csv"
        model_path = f"models/convtrans_peak/{args.forecast_type}/{args.province}"
        scaler_path = f"models/scalers/convtrans_peak/{args.forecast_type}/{args.province}"
        
        results, metrics = perform_interval_forecast(
            data_path=data_path,
            forecast_type=args.forecast_type,
            province=args.province,
            time_point=args.forecast_date,
            n_intervals=args.historical_days,
            model_path=model_path,
            scaler_path=scaler_path,
            device=device,
            fix_nan=not args.no_fix_nan
        )
        
        if results is not None:
            print("\n预测结果摘要:")
            print(results.head())
            print("\n预测统计:")
            print(f"  预测点数: {len(results)}")
            print(f"  平均预测值: {results['predicted'].mean():.2f}")
            print(f"  最小预测值: {results['predicted'].min():.2f}")
            print(f"  最大预测值: {results['predicted'].max():.2f}")
            print(f"  平均区间宽度: {(results['upper_bound'] - results['lower_bound']).mean():.2f}")
            
            # 保存预测结果
            if args.save:
                os.makedirs('results/interval_forecast', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = f"results/interval_forecast/{args.province}_{args.forecast_type}_{args.forecast_date}_{timestamp}.csv"
                results.to_csv(csv_path)
                print(f"预测结果已保存至: {csv_path}")
            
            print("优化版区间预测成功完成!")
        else:
            print("错误: 未能生成预测结果。")
    except Exception as e:
        traceback.print_exc()
        print(f"区间预测发生错误: {str(e)}")

def main():
    """主函数，支持命令行调用"""
    parser = argparse.ArgumentParser(description='优化版区间预测脚本')
    parser.add_argument('--province', type=str, required=True, help='省份名称')
    parser.add_argument('--forecast_date', type=str, required=True, help='预测日期，格式为YYYY-MM-DD')
    parser.add_argument('--forecast_type', type=str, default='load', help='预测类型: load, pv, wind')
    parser.add_argument('--confidence_level', type=float, default=0.9, help='置信水平，默认0.9')
    parser.add_argument('--historical_days', type=int, default=14, help='用于计算误差的历史天数，默认14天')
    parser.add_argument('--no_fix_nan', action='store_true', help='不自动修复NaN值')
    parser.add_argument('--device', type=str, default='auto', help='计算设备: cpu, cuda, auto')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 构建路径 - 使用目录路径而不是文件路径
        data_path = f"data/timeseries_{args.forecast_type}_{args.province}.csv"
        model_path = os.path.join(root_dir, f"models/convtrans_peak/{args.forecast_type}/{args.province}")
        scaler_path = os.path.join(root_dir, f"models/scalers/convtrans_peak/{args.forecast_type}/{args.province}")
        
        # 设置设备
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        # 转换日期字符串为datetime对象
        time_point = pd.to_datetime(args.forecast_date)
        
        results, metrics = perform_interval_forecast(
            data_path=data_path,
            forecast_type=args.forecast_type,
            province=args.province,
            time_point=time_point,
            n_intervals=args.historical_days,
            model_path=model_path,
            scaler_path=scaler_path,
            device=device,
            quantiles=[0.05, 0.95] if args.confidence_level == 0.9 else [(1-args.confidence_level)/2, 1-(1-args.confidence_level)/2],
            rolling=True,
            seq_length=96,
            peak_hours=(9, 20),
            valley_hours=(0, 6),
            fix_nan=not args.no_fix_nan
        )
        
        # 输出结果
        print(results)
        
        # 保存到CSV
        output_dir = os.path.join(root_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"optimized_interval_forecast_{args.forecast_type}_{args.province}_{args.forecast_date}.csv")
        results.to_csv(output_file)
        logger.info(f"结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"执行过程中出错: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()