#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强的区间预测模型 - 集成新能源预测与场景识别
结合天气感知的光伏和风电预测，实现综合场景识别和负荷区间预测
主要特色：
1. 同时预测负荷、光伏、风电的天气感知模型
2. 新能源大发判断算法
3. 增强的场景识别（包含新能源大发状态）
4. 基于综合场景的区间预测优化
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json
import traceback

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 导入项目相关模块
from scripts.forecast.interval_forecast_fixed import perform_interval_forecast
from scripts.forecast.day_ahead_forecast import perform_weather_aware_day_ahead_forecast
from utils.weather_scenario_classifier import WeatherScenarioClassifier
from utils.scenario_matcher import ScenarioMatcher

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_interval_forecast_with_renewables")

class RenewableEnergyPredictor:
    """新能源预测器 - 整合PV和Wind的天气感知预测"""
    
    def __init__(self, province):
        self.province = province
        self.pv_model_dir = f"models/convtrans_weather/pv/{province}"
        self.wind_model_dir = f"models/convtrans_weather/wind/{province}"
        self.pv_data_path = f"data/timeseries_pv_weather_{province}.csv"
        self.wind_data_path = f"data/timeseries_wind_weather_{province}.csv"
        
        # 新能源大发阈值配置
        self.high_output_thresholds = {
            'pv': {
                'percentile': 80,  # 光伏出力超过历史80%分位数认为是大发
                'absolute_min': 200,  # 绝对最小值（MW）
                'weather_conditions': {
                    'radiation_min': 600,  # 最小太阳辐射 W/m²
                    'cloud_cover_max': 30,  # 最大云量百分比
                    'temperature_optimal_range': [15, 30]  # 最优温度范围°C
                }
            },
            'wind': {
                'percentile': 75,  # 风电出力超过历史75%分位数认为是大发
                'absolute_min': 100,  # 绝对最小值（MW）
                'weather_conditions': {
                    'wind_speed_min': 6,  # 最小风速 m/s
                    'wind_speed_max': 15,  # 最大有效风速 m/s
                    'sustained_hours': 2  # 持续时间（小时）
                }
            }
        }
        
        # 天气特征配置
        self.weather_features = {
            'pv': ['temperature', 'humidity', 'precipitation', 'solar_radiation'],
            'wind': ['wind_speed', 'temperature', 'humidity']
        }
        
    def predict_renewables(self, forecast_date, forecast_end_date=None, historical_days=7):
        """
        预测新能源出力（PV和Wind）
        
        Args:
            forecast_date: 预测开始日期
            forecast_end_date: 预测结束日期（可选）
            historical_days: 历史数据天数
            
        Returns:
            dict: 包含PV和Wind预测结果的字典
        """
        results = {
            'pv': {'predictions': None, 'high_output_periods': [], 'error': None, 'status': 'unknown'},
            'wind': {'predictions': None, 'high_output_periods': [], 'error': None, 'status': 'unknown'},
            'combined_high_output': {'periods': [], 'analysis': None}
        }
        
        # 预测光伏
        if os.path.exists(self.pv_data_path) and os.path.exists(self.pv_model_dir):
            try:
                logger.info(f"开始预测光伏出力: {forecast_date}")
                pv_result = perform_weather_aware_day_ahead_forecast(
                    data_path=self.pv_data_path,
                    forecast_date=forecast_date,
                    forecast_end_date=forecast_end_date,
                    weather_features=self.weather_features['pv'],
                    dataset_id=self.province,
                    forecast_type='pv',
                    historical_days=historical_days
                )
                
                if isinstance(pv_result, dict) and 'predictions' in pv_result:
                    pv_predictions_raw = pv_result['predictions']
                    
                    # 检查预测结果是否为空
                    if not pv_predictions_raw or len(pv_predictions_raw) == 0:
                        logger.warning(f"光伏预测返回空结果，可能是目标日期 {forecast_date} 的数据缺失")
                        results['pv']['status'] = 'no_data'
                        results['pv']['error'] = f"目标日期 {forecast_date} 的光伏数据缺失，无法进行预测"
                        results['pv']['predictions'] = []
                        results['pv']['quality_assessment'] = {
                            'night_zero_rate': 0.0,
                            'day_valid_rate': 0.0,
                            'total_points': 0,
                            'night_points': 0,
                            'day_points': 0
                        }
                    else:
                        # 应用光伏夜间约束
                        pv_predictions = self._apply_pv_night_constraints(
                            pv_predictions_raw, 
                            pv_result.get('timestamps', [])
                        )
                        
                        results['pv']['predictions'] = pv_predictions
                        results['pv']['status'] = 'success'
                        
                        # 分析光伏大发时段
                        results['pv']['high_output_periods'] = self._analyze_high_output_periods(
                            pv_predictions, 'pv', pv_result.get('timestamps', [])
                        )
                        
                        # 计算光伏预测质量评估
                        pv_quality = self._assess_pv_prediction_quality(pv_predictions, pv_result.get('timestamps', []))
                        results['pv']['quality_assessment'] = pv_quality
                        
                        logger.info(f"光伏预测完成，识别到 {len(results['pv']['high_output_periods'])} 个大发时段")
                        logger.info(f"光伏预测质量评估: 夜间零值率={pv_quality['night_zero_rate']:.1%}, 白天有效预测率={pv_quality['day_valid_rate']:.1%}")
                else:
                    logger.warning("光伏预测结果格式不正确")
                    results['pv']['status'] = 'format_error'
                    results['pv']['error'] = "预测结果格式不正确"
                    
            except Exception as e:
                logger.error(f"光伏预测失败: {e}")
                results['pv']['error'] = str(e)
                results['pv']['status'] = 'failed'
        else:
            logger.warning(f"光伏模型或数据不存在: 模型={os.path.exists(self.pv_model_dir)}, 数据={os.path.exists(self.pv_data_path)}")
            results['pv']['error'] = f"模型或数据文件不存在: {self.pv_model_dir}, {self.pv_data_path}"
            results['pv']['status'] = 'missing_files'
        
        # 预测风电
        if os.path.exists(self.wind_data_path) and os.path.exists(self.wind_model_dir):
            try:
                logger.info(f"开始预测风电出力: {forecast_date}")
                wind_result = perform_weather_aware_day_ahead_forecast(
                    data_path=self.wind_data_path,
                    forecast_date=forecast_date,
                    forecast_end_date=forecast_end_date,
                    weather_features=self.weather_features['wind'],
                    dataset_id=self.province,
                    forecast_type='wind',
                    historical_days=historical_days
                )
                
                if isinstance(wind_result, dict) and 'predictions' in wind_result:
                    wind_predictions = wind_result['predictions']
                    
                    # 检查预测结果是否为空
                    if not wind_predictions or len(wind_predictions) == 0:
                        logger.warning(f"风电预测返回空结果，可能是目标日期 {forecast_date} 的数据缺失")
                        results['wind']['status'] = 'no_data'
                        results['wind']['error'] = f"目标日期 {forecast_date} 的风电数据缺失，无法进行预测"
                        results['wind']['predictions'] = []
                        results['wind']['quality_assessment'] = {
                            'valid_rate': 0.0,
                            'avg_output': 0.0,
                            'total_points': 0,
                            'valid_points': 0
                        }
                    else:
                        # 应用风电合理性约束
                        wind_predictions = self._apply_wind_constraints(wind_predictions)
                        
                        results['wind']['predictions'] = wind_predictions
                        results['wind']['status'] = 'success'
                        
                        # 分析风电大发时段
                        results['wind']['high_output_periods'] = self._analyze_high_output_periods(
                            wind_predictions, 'wind', wind_result.get('timestamps', [])
                        )
                        
                        # 计算风电预测质量评估
                        wind_quality = self._assess_wind_prediction_quality(wind_predictions)
                        results['wind']['quality_assessment'] = wind_quality
                        
                        logger.info(f"风电预测完成，识别到 {len(results['wind']['high_output_periods'])} 个大发时段")
                        logger.info(f"风电预测质量评估: 有效预测率={wind_quality['valid_rate']:.1%}, 平均出力={wind_quality['avg_output']:.2f}MW")
                else:
                    logger.warning("风电预测结果格式不正确")
                    results['wind']['status'] = 'format_error'
                    results['wind']['error'] = "预测结果格式不正确"
                    
            except Exception as e:
                logger.error(f"风电预测失败: {e}")
                results['wind']['error'] = str(e)
                results['wind']['status'] = 'failed'
        else:
            logger.warning(f"风电模型或数据不存在: 模型={os.path.exists(self.wind_model_dir)}, 数据={os.path.exists(self.wind_data_path)}")
            results['wind']['error'] = f"模型或数据文件不存在: {self.wind_model_dir}, {self.wind_data_path}"
            results['wind']['status'] = 'missing_files'
        
        # 分析综合新能源大发情况
        results['combined_high_output'] = self._analyze_combined_high_output(results)
        
        return results
    
    def _apply_pv_night_constraints(self, predictions, timestamps=None):
        """
        对光伏预测应用夜间约束
        
        Args:
            predictions: 原始预测值列表
            timestamps: 时间戳列表（可选）
            
        Returns:
            应用约束后的预测值列表
        """
        if not predictions:
            return predictions
            
        constrained_predictions = []
        night_adjustments = 0
        
        for i, pred in enumerate(predictions):
            # 确定时间戳
            if timestamps and i < len(timestamps):
                if isinstance(timestamps[i], str):
                    timestamp = pd.to_datetime(timestamps[i])
                else:
                    timestamp = timestamps[i]
                hour = timestamp.hour
            else:
                # 如果没有时间戳，假设从00:00开始，每15分钟一个点
                hour = (i * 15 // 60) % 24
            
            # 夜间时段：18:00-06:00
            is_night = (hour >= 18) or (hour <= 6)
            
            if is_night:
                # 夜间强制设为0
                if pred > 1.0:  # 记录被修正的预测
                    night_adjustments += 1
                constrained_predictions.append(0.0)
            else:
                # 白天时段，确保非负
                constrained_predictions.append(max(0.0, pred))
        
        if night_adjustments > 0:
            logger.info(f"光伏夜间约束: 修正了 {night_adjustments} 个夜间非零预测值")
        
        return constrained_predictions
    
    def _apply_wind_constraints(self, predictions):
        """
        对风电预测应用合理性约束
        
        Args:
            predictions: 原始预测值列表
            
        Returns:
            应用约束后的预测值列表
        """
        if not predictions:
            return predictions
            
        constrained_predictions = []
        negative_adjustments = 0
        
        for pred in predictions:
            # 风电预测值不能为负
            if pred < 0:
                negative_adjustments += 1
                constrained_predictions.append(0.0)
            else:
                constrained_predictions.append(pred)
        
        if negative_adjustments > 0:
            logger.info(f"风电约束: 修正了 {negative_adjustments} 个负值预测")
        
        return constrained_predictions
    
    def _assess_pv_prediction_quality(self, predictions, timestamps=None):
        """
        评估光伏预测质量
        
        Args:
            predictions: 预测值列表
            timestamps: 时间戳列表（可选）
            
        Returns:
            质量评估结果字典
        """
        if not predictions:
            return {'night_zero_rate': 0.0, 'day_valid_rate': 0.0, 'total_points': 0}
        
        night_points = 0
        night_zero_points = 0
        day_points = 0
        day_valid_points = 0
        
        for i, pred in enumerate(predictions):
            # 确定时间戳
            if timestamps and i < len(timestamps):
                if isinstance(timestamps[i], str):
                    timestamp = pd.to_datetime(timestamps[i])
                else:
                    timestamp = timestamps[i]
                hour = timestamp.hour
            else:
                # 如果没有时间戳，假设从00:00开始，每15分钟一个点
                hour = (i * 15 // 60) % 24
            
            # 夜间时段：18:00-06:00
            is_night = (hour >= 18) or (hour <= 6)
            
            if is_night:
                night_points += 1
                if pred <= 0.1:  # 接近零
                    night_zero_points += 1
            else:
                day_points += 1
                if pred >= 0 and not np.isnan(pred) and not np.isinf(pred):
                    day_valid_points += 1
        
        night_zero_rate = night_zero_points / night_points if night_points > 0 else 1.0
        day_valid_rate = day_valid_points / day_points if day_points > 0 else 0.0
        
        return {
            'night_zero_rate': night_zero_rate,
            'day_valid_rate': day_valid_rate,
            'total_points': len(predictions),
            'night_points': night_points,
            'day_points': day_points
        }
    
    def _assess_wind_prediction_quality(self, predictions):
        """
        评估风电预测质量
        
        Args:
            predictions: 预测值列表
            
        Returns:
            质量评估结果字典
        """
        if not predictions:
            return {'valid_rate': 0.0, 'avg_output': 0.0, 'total_points': 0}
        
        valid_predictions = []
        for pred in predictions:
            if pred >= 0 and not np.isnan(pred) and not np.isinf(pred):
                valid_predictions.append(pred)
        
        valid_rate = len(valid_predictions) / len(predictions) if predictions else 0.0
        avg_output = np.mean(valid_predictions) if valid_predictions else 0.0
        
        return {
            'valid_rate': valid_rate,
            'avg_output': avg_output,
            'total_points': len(predictions),
            'valid_points': len(valid_predictions)
        }
    
    def _analyze_high_output_periods(self, predictions, energy_type, timestamps=None):
        """分析新能源大发时段"""
        if not predictions or len(predictions) == 0:
            return []
        
        predictions = np.array(predictions)
        threshold_config = self.high_output_thresholds[energy_type]
        
        # 计算大发阈值
        percentile_threshold = np.percentile(predictions, threshold_config['percentile'])
        absolute_threshold = threshold_config['absolute_min']
        threshold = max(percentile_threshold, absolute_threshold)
        
        # 识别大发时段
        high_output_mask = predictions >= threshold
        high_output_periods = []
        
        # 将连续的大发时段合并
        if timestamps is None:
            timestamps = list(range(len(predictions)))
        
        # 确保timestamps是可JSON序列化的字符串格式
        safe_timestamps = []
        for ts in timestamps:
            if isinstance(ts, (pd.Timestamp, datetime)):
                safe_timestamps.append(ts.strftime('%Y-%m-%dT%H:%M:%S'))
            elif hasattr(ts, 'isoformat'):
                safe_timestamps.append(ts.isoformat())
            else:
                safe_timestamps.append(str(ts))
        
        start_idx = None
        for i, is_high in enumerate(high_output_mask):
            if is_high and start_idx is None:
                start_idx = i
            elif not is_high and start_idx is not None:
                # 结束一个大发时段 - 确保所有值都是JSON可序列化的
                period = {
                    'start_time': safe_timestamps[start_idx] if len(safe_timestamps) > start_idx else str(start_idx),
                    'end_time': safe_timestamps[i-1] if len(safe_timestamps) > i-1 else str(i-1),
                    'duration_hours': float((i - start_idx) * 0.25),  # 确保是float类型
                    'avg_output': float(np.mean(predictions[start_idx:i])),
                    'max_output': float(np.max(predictions[start_idx:i])),
                    'threshold': float(threshold)
                }
                
                # 进一步确保没有NaN或无穷大值
                for key, value in period.items():
                    if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
                        period[key] = 0.0
                    elif hasattr(value, 'item'):  # 处理numpy标量
                        period[key] = value.item()
                
                high_output_periods.append(period)
                start_idx = None
        
        # 处理最后一个时段
        if start_idx is not None:
            period = {
                'start_time': safe_timestamps[start_idx] if len(safe_timestamps) > start_idx else str(start_idx),
                'end_time': safe_timestamps[-1] if safe_timestamps else str(len(predictions)-1),
                'duration_hours': float((len(predictions) - start_idx) * 0.25),
                'avg_output': float(np.mean(predictions[start_idx:])),
                'max_output': float(np.max(predictions[start_idx:])),
                'threshold': float(threshold)
            }
            
            # 进一步确保没有NaN或无穷大值
            for key, value in period.items():
                if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
                    period[key] = 0.0
                elif hasattr(value, 'item'):  # 处理numpy标量
                    period[key] = value.item()
            
            high_output_periods.append(period)
        
        logger.info(f"{energy_type} 大发分析: 阈值={threshold:.2f}, 识别到 {len(high_output_periods)} 个时段")
        return high_output_periods
    
    def _analyze_combined_high_output(self, results):
        """分析综合新能源大发情况"""
        combined_analysis = {
            'periods': [],
            'analysis': {
                'total_renewable_high_periods': 0,
                'pv_only_periods': 0,
                'wind_only_periods': 0,
                'both_high_periods': 0,
                'peak_combined_output_hour': None,
                'renewable_penetration_level': 'unknown'
            }
        }
        
        pv_periods = results['pv']['high_output_periods']
        wind_periods = results['wind']['high_output_periods']
        
        # 统计不同类型的大发时段
        combined_analysis['analysis']['pv_only_periods'] = len(pv_periods)
        combined_analysis['analysis']['wind_only_periods'] = len(wind_periods)
        
        # 寻找重叠的大发时段（PV和Wind同时大发）
        both_high_periods = []
        for pv_period in pv_periods:
            for wind_period in wind_periods:
                if self._periods_overlap(pv_period, wind_period):
                    # 确保overlap_start和overlap_end也是JSON可序列化的
                    try:
                        pv_start = pv_period['start_time']
                        pv_end = pv_period['end_time']
                        wind_start = wind_period['start_time']
                        wind_end = wind_period['end_time']
                        
                        # 如果是字符串格式，直接比较；否则转换为字符串
                        if isinstance(pv_start, str) and isinstance(wind_start, str):
                            overlap_start = max(pv_start, wind_start)
                            overlap_end = min(pv_end, wind_end)
                        else:
                            # 转换为可比较的格式再转回字符串
                            pv_start_dt = pd.to_datetime(pv_start) if not isinstance(pv_start, pd.Timestamp) else pv_start
                            wind_start_dt = pd.to_datetime(wind_start) if not isinstance(wind_start, pd.Timestamp) else wind_start
                            pv_end_dt = pd.to_datetime(pv_end) if not isinstance(pv_end, pd.Timestamp) else pv_end
                            wind_end_dt = pd.to_datetime(wind_end) if not isinstance(wind_end, pd.Timestamp) else wind_end
                            
                            overlap_start = max(pv_start_dt, wind_start_dt).strftime('%Y-%m-%dT%H:%M:%S')
                            overlap_end = min(pv_end_dt, wind_end_dt).strftime('%Y-%m-%dT%H:%M:%S')
                        
                        both_high_periods.append({
                            'pv_period': pv_period,
                            'wind_period': wind_period,
                            'overlap_start': overlap_start,
                            'overlap_end': overlap_end,
                            'combined_significance': 'high'
                        })
                    except Exception as e:
                        logger.warning(f"处理重叠时段时出错: {e}")
                        # 降级处理：只包含基本信息
                        both_high_periods.append({
                            'pv_period': pv_period,
                            'wind_period': wind_period,
                            'overlap_start': str(pv_period['start_time']),
                            'overlap_end': str(pv_period['end_time']),
                            'combined_significance': 'high'
                        })
        
        combined_analysis['analysis']['both_high_periods'] = len(both_high_periods)
        combined_analysis['analysis']['total_renewable_high_periods'] = len(pv_periods) + len(wind_periods)
        combined_analysis['periods'] = both_high_periods
        
        # 判断新能源渗透率水平
        if both_high_periods:
            combined_analysis['analysis']['renewable_penetration_level'] = 'high'
        elif len(pv_periods) > 3 or len(wind_periods) > 3:
            combined_analysis['analysis']['renewable_penetration_level'] = 'medium'
        else:
            combined_analysis['analysis']['renewable_penetration_level'] = 'low'
        
        return combined_analysis
    
    def _periods_overlap(self, period1, period2):
        """检查两个时段是否重叠"""
        try:
            # 将时间转换为可比较的格式
            if isinstance(period1['start_time'], str):
                start1 = pd.to_datetime(period1['start_time'])
                end1 = pd.to_datetime(period1['end_time'])
            else:
                start1 = period1['start_time']
                end1 = period1['end_time']
            
            if isinstance(period2['start_time'], str):
                start2 = pd.to_datetime(period2['start_time'])
                end2 = pd.to_datetime(period2['end_time'])
            else:
                start2 = period2['start_time']
                end2 = period2['end_time']
            
            return not (end1 < start2 or end2 < start1)
        except Exception as e:
            logger.warning(f"时段重叠检查失败: {e}")
            return False

class EnhancedScenarioClassifier:
    """增强的场景分类器 - 包含新能源大发判断"""
    
    def __init__(self):
        self.weather_classifier = WeatherScenarioClassifier()
        self.base_scenarios = {
            'renewable_surge': {
                'name': '新能源大发',
                'description': '光伏或风电出力显著高于平均水平，系统灵活性需求增加',
                'uncertainty_multiplier': 1.3,
                'risk_level': 'medium',
                'load_impact': 'potential_reduction'
            },
            'renewable_dual_surge': {
                'name': '双重新能源大发',
                'description': '光伏和风电同时大发，系统需要强灵活性调节',
                'uncertainty_multiplier': 1.5,
                'risk_level': 'medium-high',
                'load_impact': 'significant_reduction'
            },
            'renewable_low_weather_extreme': {
                'name': '新能源低发+极端天气',
                'description': '新能源出力低且天气极端，负荷预测不确定性最高',
                'uncertainty_multiplier': 2.8,
                'risk_level': 'high',
                'load_impact': 'increased'
            },
            'renewable_high_load_peak': {
                'name': '新能源大发+负荷高峰',
                'description': '新能源大发与负荷高峰重叠，系统调节复杂',
                'uncertainty_multiplier': 1.4,
                'risk_level': 'medium',
                'load_impact': 'complex_interaction'
            }
        }
    
    def classify_enhanced_scenario(self, weather_data, renewable_data, load_features=None):
        """
        增强的场景分类，考虑天气、新能源和负荷的综合情况
        
        Args:
            weather_data: 天气数据
            renewable_data: 新能源预测数据
            load_features: 负荷特征（可选）
            
        Returns:
            dict: 增强的场景信息
        """
        # 检查weather_data格式并进行适配
        if isinstance(weather_data, dict):
            # 如果weather_data已经是场景信息（包含name, risk_level等），直接使用
            if 'name' in weather_data and 'risk_level' in weather_data:
                weather_scenario = weather_data
                logger.info(f"使用已处理的天气场景信息: {weather_scenario.get('name', 'unknown')}")
            # 如果包含原始天气特征，调用基础分类器
            elif any(key in weather_data for key in ['temperature', 'humidity', 'wind_speed', 'precipitation', 'radiation']):
                weather_scenario = self.weather_classifier.identify_scenario(weather_data)
                logger.info(f"基于原始天气数据识别场景: {weather_scenario.get('name', 'unknown')}")
            else:
                # 如果数据格式不正确，使用默认场景
                logger.warning(f"天气数据格式不正确，使用默认场景。数据内容: {list(weather_data.keys()) if weather_data else 'empty'}")
                weather_scenario = {
                    'name': '温和正常',
                    'risk_level': 'low',
                    'uncertainty_multiplier': 1.0,
                    'description': '默认天气场景（数据格式问题）'
                }
        else:
            # 如果不是字典或为空，使用默认场景
            logger.warning(f"天气数据不是字典格式或为空，使用默认场景。数据类型: {type(weather_data)}")
            weather_scenario = {
                'name': '温和正常',
                'risk_level': 'low',
                'uncertainty_multiplier': 1.0,
                'description': '默认天气场景（数据缺失）'
            }
        
        # 分析新能源状态
        renewable_status = self._analyze_renewable_status(renewable_data)
        
        # 综合分析确定最终场景
        enhanced_scenario = self._determine_enhanced_scenario(
            weather_scenario, renewable_status, load_features
        )
        
        return {
            'enhanced_scenario': enhanced_scenario,
            'weather_scenario': weather_scenario,
            'renewable_status': renewable_status,
            'composite_risk_level': self._calculate_composite_risk(
                weather_scenario, renewable_status
            ),
            'uncertainty_adjustment': self._calculate_uncertainty_adjustment(
                weather_scenario, renewable_status
            )
        }
    
    def _analyze_renewable_status(self, renewable_data):
        """分析新能源状态"""
        if not renewable_data:
            return {
                'pv_status': 'unknown',
                'wind_status': 'unknown',
                'combined_status': 'normal_output',
                'high_output_periods': 0,
                'renewable_penetration': 'low'
            }
        
        pv_data = renewable_data.get('pv', {})
        wind_data = renewable_data.get('wind', {})
        
        # 判断PV状态 - 基于预测成功状态和高出力时段
        pv_status_flag = pv_data.get('status', 'unknown')
        if pv_status_flag == 'success':
            pv_high_periods = len(pv_data.get('high_output_periods', []))
            if pv_high_periods >= 3:
                pv_status = 'high_output'
            elif pv_high_periods >= 1:
                pv_status = 'moderate_output'
            else:
                pv_status = 'normal_output'
        elif pv_status_flag == 'missing_files':
            pv_status = 'model_unavailable'
        elif pv_status_flag == 'failed':
            pv_status = 'prediction_failed'
        else:
            pv_status = 'unknown'
        
        # 判断Wind状态 - 基于预测成功状态和高出力时段
        wind_status_flag = wind_data.get('status', 'unknown')
        if wind_status_flag == 'success':
            wind_high_periods = len(wind_data.get('high_output_periods', []))
            if wind_high_periods >= 4:
                wind_status = 'high_output'
            elif wind_high_periods >= 2:
                wind_status = 'moderate_output'
            else:
                wind_status = 'normal_output'
        elif wind_status_flag == 'missing_files':
            wind_status = 'model_unavailable'
        elif wind_status_flag == 'failed':
            wind_status = 'prediction_failed'
        else:
            wind_status = 'unknown'
        
        # 综合状态 - 只有在两个预测都成功时才进行综合分析
        if pv_status_flag == 'success' and wind_status_flag == 'success':
            pv_high_periods = len(pv_data.get('high_output_periods', []))
            wind_high_periods = len(wind_data.get('high_output_periods', []))
            total_high_periods = pv_high_periods + wind_high_periods
            
            if total_high_periods >= 5:
                combined_status = 'dual_high_output'
                renewable_penetration = 'high'
            elif total_high_periods >= 3:
                combined_status = 'moderate_output'
                renewable_penetration = 'medium'
            else:
                combined_status = 'normal_output'
                renewable_penetration = 'low'
        elif pv_status_flag == 'success' or wind_status_flag == 'success':
            # 只有一个成功，基于成功的那个进行评估
            if pv_status_flag == 'success':
                high_periods = len(pv_data.get('high_output_periods', []))
            else:
                high_periods = len(wind_data.get('high_output_periods', []))
            
            if high_periods >= 3:
                combined_status = 'single_high_output'
                renewable_penetration = 'medium'
            else:
                combined_status = 'normal_output'
                renewable_penetration = 'low'
            total_high_periods = high_periods
        else:
            # 两个都失败或不可用
            combined_status = 'prediction_unavailable'
            renewable_penetration = 'unknown'
            total_high_periods = 0
        
        status = {
            'pv_status': pv_status,
            'wind_status': wind_status,
            'combined_status': combined_status,
            'high_output_periods': total_high_periods,
            'renewable_penetration': renewable_penetration,
            'pv_quality': pv_data.get('quality_assessment', {}),
            'wind_quality': wind_data.get('quality_assessment', {})
        }
        
        return status
    
    def _determine_enhanced_scenario(self, weather_scenario, renewable_status, load_features):
        """确定增强场景"""
        weather_risk = weather_scenario.get('risk_level', 'low')
        renewable_combined = renewable_status.get('combined_status', 'normal_output')
        
        # 规则引擎决定场景
        if renewable_combined == 'dual_high_output':
            if weather_risk == 'high':
                return self.base_scenarios['renewable_low_weather_extreme']
            else:
                return self.base_scenarios['renewable_dual_surge']
        
        elif renewable_combined == 'single_high_output':
            if load_features and load_features.get('is_peak_hour', False):
                return self.base_scenarios['renewable_high_load_peak']
            else:
                return self.base_scenarios['renewable_surge']
        
        elif renewable_combined == 'normal_output' and weather_risk == 'high':
            return self.base_scenarios['renewable_low_weather_extreme']
        
        else:
            # 返回基础天气场景
            return {
                'name': weather_scenario.get('name', '标准场景'),
                'description': weather_scenario.get('description', ''),
                'uncertainty_multiplier': weather_scenario.get('uncertainty_multiplier', 1.0),
                'risk_level': weather_risk,
                'load_impact': 'normal'
            }
    
    def _calculate_composite_risk(self, weather_scenario, renewable_status):
        """计算综合风险等级"""
        weather_risk = weather_scenario.get('risk_level', 'low')
        renewable_penetration = renewable_status.get('renewable_penetration', 'low')
        
        risk_matrix = {
            ('high', 'high'): 'very_high',
            ('high', 'medium'): 'high',
            ('high', 'low'): 'medium-high',
            ('medium', 'high'): 'high',
            ('medium', 'medium'): 'medium',
            ('medium', 'low'): 'medium',
            ('low', 'high'): 'medium',
            ('low', 'medium'): 'low-medium',
            ('low', 'low'): 'low'
        }
        
        return risk_matrix.get((weather_risk, renewable_penetration), 'medium')
    
    def _calculate_uncertainty_adjustment(self, weather_scenario, renewable_status):
        """计算不确定性调整系数"""
        base_uncertainty = weather_scenario.get('uncertainty_multiplier', 1.0)
        renewable_factor = 1.0
        
        # 根据新能源状态调整不确定性
        if renewable_status.get('combined_status') == 'dual_high_output':
            renewable_factor = 1.3
        elif renewable_status.get('combined_status') == 'single_high_output':
            renewable_factor = 1.15
        
        return base_uncertainty * renewable_factor

def perform_enhanced_interval_forecast_with_renewables(
    province, forecast_date, forecast_end_date=None, confidence_level=0.9, 
    historical_days=7, enable_renewable_prediction=True
):
    """
    执行增强的区间预测，包含新能源预测和场景识别
    
    Args:
        province: 省份名称
        forecast_date: 预测开始日期
        forecast_end_date: 预测结束日期（可选）
        confidence_level: 置信水平
        historical_days: 历史数据天数
        enable_renewable_prediction: 是否启用新能源预测
        
    Returns:
        tuple: (预测结果DataFrame, 综合指标字典)
    """
    logger.info(f"===== 开始增强区间预测 (含新能源): {province}, {forecast_date} =====")
    
    try:
        # 1. 执行基础负荷区间预测
        logger.info("执行基础负荷区间预测...")
        load_data_path = f"data/timeseries_load_weather_{province}.csv"
        load_model_path = f"models/convtrans_weather/load/{province}"
        load_scaler_path = f"models/scalers/convtrans_weather/load/{province}"
        
        # 调用天气感知区间预测
        from scripts.forecast.weather_aware_interval_forecast import perform_weather_aware_interval_forecast_for_range
        
        load_results, load_metrics = perform_weather_aware_interval_forecast_for_range(
            province=province,
            forecast_type='load',
            start_date_str=forecast_date,
            end_date_str=forecast_end_date or forecast_date,
            confidence_level=confidence_level,
            historical_days=historical_days
        )
        
        # 2. 预测新能源出力（如果启用）
        renewable_results = None
        if enable_renewable_prediction:
            logger.info("执行新能源预测...")
            try:
                renewable_predictor = RenewableEnergyPredictor(province)
                renewable_results = renewable_predictor.predict_renewables(
                    forecast_date, forecast_end_date, historical_days
                )
                logger.info(f"新能源预测完成: PV状态={renewable_results.get('pv', {}).get('error', '成功')}, Wind状态={renewable_results.get('wind', {}).get('error', '成功')}")
            except Exception as e:
                logger.warning(f"新能源预测失败，将继续使用负荷预测: {e}")
                renewable_results = None
        
        # 3. 增强场景分析
        logger.info("执行增强场景分析...")
        try:
            enhanced_classifier = EnhancedScenarioClassifier()
            
            # 提取天气数据用于场景分析
            weather_data = load_metrics.get('weather_scenario', {})
            logger.info(f"从负荷预测metrics中提取的天气数据类型: {type(weather_data)}")
            logger.info(f"天气数据内容: {list(weather_data.keys()) if isinstance(weather_data, dict) else weather_data}")
            
            enhanced_scenario_info = enhanced_classifier.classify_enhanced_scenario(
                weather_data=weather_data,
                renewable_data=renewable_results,
                load_features={
                    'daily_load_mean': load_metrics.get('daily_load_mean', 0),
                    'daily_load_volatility': load_metrics.get('daily_load_volatility', 0)
                }
            )
        except Exception as e:
            logger.warning(f"增强场景分析失败，使用默认场景: {e}")
            # 创建默认的增强场景信息
            enhanced_scenario_info = {
                'enhanced_scenario': {
                    'name': '标准负荷预测',
                    'description': '默认场景（场景分析失败）',
                    'uncertainty_multiplier': 1.0,
                    'risk_level': 'medium',
                    'load_impact': 'normal'
                },
                'weather_scenario': {
                    'name': '温和正常',
                    'risk_level': 'low',
                    'uncertainty_multiplier': 1.0
                },
                'renewable_status': {
                    'pv_status': 'unknown',
                    'wind_status': 'unknown',
                    'combined_status': 'normal_output'
                },
                'composite_risk_level': 'medium',
                'uncertainty_adjustment': 1.0
            }
        
        # 4. 基于增强场景调整区间预测
        logger.info("基于增强场景调整预测区间...")
        enhanced_results = _adjust_intervals_based_on_scenario(
            load_results, enhanced_scenario_info, confidence_level
        )
        
        # 5. 整合综合指标
        comprehensive_metrics = {
            **load_metrics,
            'renewable_predictions': renewable_results,
            'enhanced_scenario': enhanced_scenario_info,
            'forecast_adjustments': {
                'scenario_based_adjustment': True,
                'renewable_integration': enable_renewable_prediction,
                'composite_risk_level': enhanced_scenario_info.get('composite_risk_level', 'unknown')
            }
        }
        
        logger.info(f"增强区间预测完成")
        logger.info(f"识别场景: {enhanced_scenario_info['enhanced_scenario']['name']}")
        logger.info(f"综合风险等级: {enhanced_scenario_info.get('composite_risk_level', 'unknown')}")
        
        return enhanced_results, comprehensive_metrics
        
    except Exception as e:
        logger.error(f"增强区间预测失败: {e}")
        traceback.print_exc()
        raise

def _adjust_intervals_based_on_scenario(load_results, enhanced_scenario_info, confidence_level):
    """基于增强场景调整预测区间"""
    if load_results is None or load_results.empty:
        return load_results
    
    # 获取场景调整系数
    uncertainty_adjustment = enhanced_scenario_info.get('uncertainty_adjustment', 1.0)
    
    # 调整区间宽度
    if 'lower_bound' in load_results.columns and 'upper_bound' in load_results.columns:
        # 计算当前区间宽度
        current_width = load_results['upper_bound'] - load_results['lower_bound']
        adjusted_width = current_width * uncertainty_adjustment
        
        # 重新计算区间边界
        midpoint = (load_results['upper_bound'] + load_results['lower_bound']) / 2
        half_width = adjusted_width / 2
        
        load_results['lower_bound'] = midpoint - half_width
        load_results['upper_bound'] = midpoint + half_width
        
        # 确保下界不为负
        load_results['lower_bound'] = np.maximum(0, load_results['lower_bound'])
        
        logger.info(f"基于场景调整区间宽度，调整系数: {uncertainty_adjustment:.3f}")
    
    return load_results

# 主函数接口
def main():
    """主函数，支持命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强区间预测脚本（含新能源）')
    parser.add_argument('--province', type=str, required=True, help='省份名称')
    parser.add_argument('--forecast_date', type=str, required=True, help='预测日期，格式为YYYY-MM-DD')
    parser.add_argument('--forecast_end_date', type=str, default=None, help='预测结束日期（可选）')
    parser.add_argument('--confidence_level', type=float, default=0.9, help='置信水平，默认0.9')
    parser.add_argument('--historical_days', type=int, default=7, help='历史数据天数，默认7天')
    parser.add_argument('--no_renewable', action='store_true', help='禁用新能源预测')
    parser.add_argument('--output_json', type=str, default=None, help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    try:
        results_df, metrics = perform_enhanced_interval_forecast_with_renewables(
            province=args.province,
            forecast_date=args.forecast_date,
            forecast_end_date=args.forecast_end_date,
            confidence_level=args.confidence_level,
            historical_days=args.historical_days,
            enable_renewable_prediction=not args.no_renewable
        )
        
        # 输出结果
        print(f"预测完成，数据行数: {len(results_df)}")
        print(f"识别场景: {metrics['enhanced_scenario']['enhanced_scenario']['name']}")
        
        # 保存JSON输出
        if args.output_json:
            output_data = {
                'status': 'success',
                'province': args.province,
                'forecast_type': 'enhanced_load_interval_with_renewables',
                'forecast_date': args.forecast_date,
                'predictions': results_df.to_dict(orient='records'),
                'metrics': metrics,
                'enhanced_features': {
                    'renewable_integration': not args.no_renewable,
                    'scenario_based_adjustment': True,
                    'composite_risk_assessment': True
                }
            }
            
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4, default=str)
            print(f"结果已保存到: {args.output_json}")
    
    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise

if __name__ == "__main__":
    main() 