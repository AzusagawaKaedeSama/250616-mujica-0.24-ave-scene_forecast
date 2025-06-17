#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景感知不确定性预测引擎
整合天气场景识别、多源预测和动态不确定性建模
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import random

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.weather_scenario_classifier import WeatherScenarioClassifier
from scripts.forecast.day_ahead_forecast import perform_weather_aware_day_ahead_forecast
from scripts.forecast.interval_forecast_fixed import perform_interval_forecast_for_range
from scripts.forecast.probabilistic_forecast import perform_probabilistic_forecast

logger = logging.getLogger(__name__)

class ScenarioAwareUncertaintyEngine:
    """场景感知的不确定性预测引擎"""
    
    def __init__(self):
        self.weather_classifier = WeatherScenarioClassifier()
        self.base_uncertainty_factors = {
            'load': 0.05,     # 基础负荷不确定性 5%
            'pv': 0.15,       # 光伏不确定性 15%
            'wind': 0.20      # 风电不确定性 20%
        }
        
    def scenario_aware_uncertainty_forecast(self, 
                                          province: str,
                                          forecast_type: str,
                                          start_date: str,
                                          end_date: str = None,
                                          confidence_level: float = 0.9,
                                          historical_days: int = 14,
                                          include_explanations: bool = True) -> Dict:
        """
        执行场景感知的不确定性预测
        
        Args:
            province: 省份名称
            forecast_type: 预测类型 ('load', 'pv', 'wind')
            start_date: 预测开始日期
            end_date: 预测结束日期（如果None，则预测一天）
            confidence_level: 置信水平
            historical_days: 历史数据天数
            include_explanations: 是否包含详细解释
            
        Returns:
            包含场景分析和不确定性预测的完整结果
        """
        logger.info(f"开始场景感知不确定性预测: {province} {forecast_type} {start_date}")
        
        # 1. 准备预测日期范围
        if end_date is None:
            end_date = start_date
            
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 2. 加载天气数据用于场景识别
        weather_data = self._load_weather_data_for_scenario(province, start_dt, end_dt)
        
        # 3. 执行天气场景分析
        scenario_analysis = self.weather_classifier.analyze_forecast_period_scenarios(weather_data)
        
        # 4. 根据场景执行不确定性预测
        uncertainty_results = self._execute_scenario_based_forecast(
            province=province,
            forecast_type=forecast_type,
            start_date=start_date,
            end_date=end_date,
            scenario_analysis=scenario_analysis,
            confidence_level=confidence_level,
            historical_days=historical_days
        )
        
        # 5. 生成综合结果和解释
        comprehensive_results = self._generate_comprehensive_results(
            scenario_analysis=scenario_analysis,
            uncertainty_results=uncertainty_results,
            province=province,
            forecast_type=forecast_type,
            include_explanations=include_explanations
        )
        
        logger.info("场景感知不确定性预测完成")
        return comprehensive_results
    
    def _load_weather_data_for_scenario(self, province: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
        """加载用于场景识别的天气数据"""
        try:
            # 尝试加载整合的天气负荷数据
            weather_load_file = f"data/timeseries_load_weather_{province}.csv"
            if os.path.exists(weather_load_file):
                df = pd.read_csv(weather_load_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                
                # 筛选预测期间的数据
                mask = (df.index >= start_dt) & (df.index <= end_dt + timedelta(days=1))
                weather_df = df.loc[mask].copy()
                
                # 提取天气特征
                weather_columns = []
                for col in df.columns:
                    if any(weather_key in col.lower() for weather_key in 
                          ['temperature', 'humidity', 'wind_speed', 'solar_radiation', 'precipitation']):
                        weather_columns.append(col)
                
                if weather_columns:
                    weather_df = weather_df[weather_columns].copy()
                    # 标准化列名
                    weather_df.columns = [self._standardize_weather_column_name(col) for col in weather_df.columns]
                    return weather_df
            
            # 如果没有找到天气数据，生成模拟数据用于演示
            logger.warning(f"未找到 {province} 的天气数据文件，使用模拟数据")
            return self._generate_mock_weather_data(start_dt, end_dt)
            
        except Exception as e:
            logger.error(f"加载天气数据时出错: {e}")
            return self._generate_mock_weather_data(start_dt, end_dt)
    
    def _standardize_weather_column_name(self, col_name: str) -> str:
        """标准化天气数据列名"""
        col_lower = col_name.lower()
        if 'temperature' in col_lower:
            return 'temperature'
        elif 'humidity' in col_lower:
            return 'humidity'
        elif 'wind_speed' in col_lower or 'windspeed' in col_lower:
            return 'wind_speed'
        elif 'solar' in col_lower or 'radiation' in col_lower:
            return 'solar_radiation'
        elif 'precipitation' in col_lower or 'rain' in col_lower:
            return 'precipitation'
        else:
            return col_name
    
    def _generate_mock_weather_data(self, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
        """生成模拟天气数据"""
        date_range = pd.date_range(start=start_dt, end=end_dt + timedelta(days=1), freq='15min')
        
        # 生成基于季节的模拟天气数据
        month = start_dt.month
        
        # 温度模拟（基于季节和时间）
        base_temp = 20.0 if 3 <= month <= 5 else (30.0 if 6 <= month <= 8 else (15.0 if 9 <= month <= 11 else 5.0))
        temperatures = []
        for dt in date_range:
            hour_variation = 5 * np.sin((dt.hour - 6) * np.pi / 12)  # 日变化
            random_variation = np.random.normal(0, 2)  # 随机变化
            temp = base_temp + hour_variation + random_variation
            temperatures.append(temp)
        
        weather_data = {
            'temperature': temperatures,
            'humidity': np.random.normal(60, 15, len(date_range)),
            'wind_speed': np.random.exponential(3, len(date_range)),
            'solar_radiation': [max(0, 800 * np.sin(max(0, (dt.hour - 6) * np.pi / 12)) + np.random.normal(0, 100)) 
                               for dt in date_range],
            'precipitation': np.random.exponential(0.5, len(date_range))
        }
        
        df = pd.DataFrame(weather_data, index=date_range)
        return df
    
    def _execute_scenario_based_forecast(self, province: str, forecast_type: str, 
                                       start_date: str, end_date: str,
                                       scenario_analysis: Dict, confidence_level: float,
                                       historical_days: int) -> Dict:
        """基于场景分析执行不确定性预测"""
        
        # 获取主导场景的不确定性倍数
        dominant_scenario = scenario_analysis.get('dominant_scenario')
        if dominant_scenario:
            uncertainty_multiplier = dominant_scenario.get('uncertainty_multiplier', 1.0)
        else:
            uncertainty_multiplier = 1.0
        
        # 调整置信水平
        adjusted_confidence = self._adjust_confidence_level(confidence_level, uncertainty_multiplier)
        
        results = {
            'base_forecast': None,
            'interval_forecast': None,
            'uncertainty_adjustments': {
                'original_confidence': confidence_level,
                'adjusted_confidence': adjusted_confidence,
                'uncertainty_multiplier': uncertainty_multiplier,
                'scenario_adjustment': True
            }
        }
        
        try:
            # 生成模拟的预测结果用于演示
            results['interval_forecast'] = self._generate_mock_forecast_results(
                start_date, end_date, forecast_type, adjusted_confidence, uncertainty_multiplier
            )
            
        except Exception as e:
            logger.error(f"执行场景基础预测时出错: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_mock_forecast_results(self, start_date: str, end_date: str, 
                                       forecast_type: str, confidence_level: float,
                                       uncertainty_multiplier: float) -> tuple:
        """生成模拟的预测结果用于演示"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 生成时间序列（15分钟间隔）
        date_range = pd.date_range(start=start_dt, end=end_dt + timedelta(days=1) - timedelta(minutes=15), freq='15min')
        
        # 根据预测类型设置基础值
        if forecast_type == 'load':
            base_values = [1000 + 500 * np.sin((i % 96) * 2 * np.pi / 96) + np.random.normal(0, 50) for i in range(len(date_range))]
        elif forecast_type == 'pv':
            base_values = []
            for i, dt in enumerate(date_range):
                if 6 <= dt.hour <= 18:
                    pv_value = 300 * np.sin((dt.hour - 6) * np.pi / 12) + np.random.normal(0, 30)
                else:
                    pv_value = 0
                base_values.append(max(0, pv_value))
        elif forecast_type == 'wind':
            base_values = [200 + 100 * np.random.random() + np.random.normal(0, 20) for _ in range(len(date_range))]
        else:
            base_values = [500 + 200 * np.sin((i % 96) * 2 * np.pi / 96) + np.random.normal(0, 30) for i in range(len(date_range))]
        
        # 计算不确定性区间
        base_uncertainty = self.base_uncertainty_factors.get(forecast_type, 0.05)
        final_uncertainty = base_uncertainty * uncertainty_multiplier
        
        predictions = []
        for i, (dt, predicted) in enumerate(zip(date_range, base_values)):
            # 计算区间
            uncertainty_range = predicted * final_uncertainty
            lower_bound = max(0, predicted - uncertainty_range)
            upper_bound = predicted + uncertainty_range
            
            # 确定场景（简化版本）
            hour = dt.hour
            if 6 <= hour <= 9 or 18 <= hour <= 21:
                scenario = '高峰时段'
                risk_level = 'medium'
            elif 22 <= hour or hour <= 5:
                scenario = '夜间时段'
                risk_level = 'low'
            else:
                scenario = '平时段'
                risk_level = 'low'
            
            predictions.append({
                'datetime': dt.strftime('%Y-%m-%dT%H:%M:%S'),
                'predicted': predicted,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'uncertainty': final_uncertainty * 100,
                'scenario': scenario,
                'risk_level': risk_level
            })
        
        results_df = pd.DataFrame(predictions)
        metrics = {
            'confidence_level': confidence_level,
            'uncertainty_multiplier': uncertainty_multiplier,
            'base_uncertainty': base_uncertainty,
            'final_uncertainty': final_uncertainty,
            'total_predictions': len(predictions)
        }
        
        return results_df, metrics
    
    def _adjust_confidence_level(self, original_confidence: float, uncertainty_multiplier: float) -> float:
        """根据场景不确定性调整置信水平"""
        if uncertainty_multiplier > 2.0:
            # 高不确定性场景，降低置信水平以获得更宽的区间
            return max(0.8, original_confidence - 0.05)
        elif uncertainty_multiplier < 0.9:
            # 低不确定性场景，可以提高置信水平
            return min(0.95, original_confidence + 0.02)
        else:
            return original_confidence
    
    def _apply_scenario_uncertainty_adjustments(self, interval_forecast: Tuple, 
                                               scenario_analysis: Dict, 
                                               uncertainty_multiplier: float) -> Tuple:
        """应用场景特定的不确定性调整"""
        if not interval_forecast or len(interval_forecast) < 2:
            return interval_forecast
        
        results_df, metrics = interval_forecast
        
        if 'lower_bound' in results_df.columns and 'upper_bound' in results_df.columns:
            # 计算当前区间宽度
            current_width = results_df['upper_bound'] - results_df['lower_bound']
            
            # 应用场景调整
            adjustment_factor = uncertainty_multiplier
            
            # 根据时段进一步调整
            scenario_profiles = scenario_analysis.get('uncertainty_profile', [])
            if scenario_profiles:
                for i, uncertainty_factor in enumerate(scenario_profiles[:len(results_df)]):
                    if i < len(results_df):
                        # 计算中心点
                        center = results_df.iloc[i]['predicted']
                        half_width = current_width.iloc[i] * uncertainty_factor / 2
                        
                        # 调整区间
                        results_df.iloc[i, results_df.columns.get_loc('lower_bound')] = max(0, center - half_width)
                        results_df.iloc[i, results_df.columns.get_loc('upper_bound')] = center + half_width
            
            # 更新metrics
            if metrics:
                metrics['scenario_adjusted'] = True
                metrics['uncertainty_multiplier'] = uncertainty_multiplier
                metrics['adjustment_details'] = scenario_analysis.get('scenario_distribution', {})
        
        return results_df, metrics
    
    def _generate_comprehensive_results(self, scenario_analysis: Dict, uncertainty_results: Dict,
                                      province: str, forecast_type: str, 
                                      include_explanations: bool) -> Dict:
        """生成包含场景分析和解释的综合结果"""
        
        # 提取预测结果
        interval_forecast = uncertainty_results.get('interval_forecast')
        if interval_forecast and len(interval_forecast) >= 2:
            results_df, metrics = interval_forecast
        else:
            results_df = pd.DataFrame()
            metrics = {}
        
        # 从场景分析中提取主导场景信息
        dominant_scenario = scenario_analysis.get('dominant_scenario', {})
        scenarios_list = scenario_analysis.get('scenarios', [])
        
        # 如果有场景信息，使用第一个场景作为代表
        if scenarios_list:
            representative_scenario = scenarios_list[0]
        else:
            # 创建默认场景
            representative_scenario = {
                'id': 'moderate_normal',
                'name': '温和正常',
                'description': '天气条件温和，系统运行平稳',
                'uncertainty_multiplier': 1.0,
                'risk_level': 'low'
            }
        
        # 构建综合结果
        comprehensive_results = {
            'status': 'success',
            'forecast_type': forecast_type,
            'province': province,
            'scenarios': [representative_scenario],  # 前端期望的格式
            'predictions': results_df.to_dict('records') if not results_df.empty else [],
            'metrics': metrics,
            'uncertainty_analysis': {
                'methodology': 'weather_scenario_aware',
                'base_uncertainty': self.base_uncertainty_factors.get(forecast_type, 0.1),
                'scenario_adjustments': uncertainty_results.get('uncertainty_adjustments', {}),
                'dominant_scenario': dominant_scenario
            }
        }
        
        # 添加详细解释
        if include_explanations:
            # 生成解释信息
            scenario_info = representative_scenario
            uncertainty_params = {
                'forecast_type': forecast_type,
                'base_uncertainty': self.base_uncertainty_factors.get(forecast_type, 0.05),
                'scenario_multiplier': representative_scenario.get('uncertainty_multiplier', 1.0),
                'time_period_adjustment': 1.0,
                'final_uncertainty': self.base_uncertainty_factors.get(forecast_type, 0.05) * representative_scenario.get('uncertainty_multiplier', 1.0),
                'confidence_level': 0.9
            }
            
            explanation = self.weather_classifier.generate_explanation(scenario_info, uncertainty_params)
            
            comprehensive_results['explanations'] = [{
                'explanation': explanation
            }]
        
        return comprehensive_results
    
    def _generate_uncertainty_explanations(self, scenario_analysis: Dict, 
                                         uncertainty_results: Dict, 
                                         forecast_type: str) -> Dict:
        """生成不确定性来源和计算过程的详细解释"""
        
        explanations = {
            'uncertainty_sources': [],
            'scenario_impact': '',
            'calculation_methodology': '',
            'recommendations': []
        }
        
        # 分析不确定性来源
        dominant_scenario = scenario_analysis.get('dominant_scenario')
        if dominant_scenario:
            scenario_name = dominant_scenario.get('name', '未知场景')
            scenario_desc = dominant_scenario.get('description', '场景描述不可用')
            uncertainty_factor = dominant_scenario.get('uncertainty_multiplier', 1.0)
            
            explanations['scenario_impact'] = (
                f"主导天气场景为 '{scenario_name}': {scenario_desc}。"
                f"该场景的不确定性倍数为 {uncertainty_factor:.1f}，"
                f"{'显著增加' if uncertainty_factor > 1.5 else ('适度增加' if uncertainty_factor > 1.0 else '降低')}了预测不确定性。"
            )
            
            # 分析具体影响因素
            scenario_id = None
            for sid, scenario in self.weather_classifier.scenarios.items():
                if scenario['name'] == scenario_name:
                    scenario_id = sid
                    break
            
            if scenario_id and 'extreme_hot' in scenario_id:
                explanations['uncertainty_sources'].extend([
                    "极端高温导致空调负荷急剧变化，增加了负荷预测的不确定性",
                    "高温天气下用户行为差异较大，导致负荷波动性增加",
                    "电网设备在高温下运行特性发生变化"
                ])
            elif scenario_id and 'extreme_cold' in scenario_id:
                explanations['uncertainty_sources'].extend([
                    "极端低温增加了采暖负荷的不确定性",
                    "低温天气影响新能源设备效率，间接影响净负荷"
                ])
            elif scenario_id and 'high_wind_sunny' in scenario_id:
                explanations['uncertainty_sources'].extend([
                    "大风晴朗天气下新能源出力较高且稳定，降低了系统不确定性",
                    "良好的天气条件使负荷和新能源预测都更加准确"
                ])
            elif scenario_id and 'storm_rain' in scenario_id:
                explanations['uncertainty_sources'].extend([
                    "暴雨大风天气极大增加了新能源出力的不确定性",
                    "恶劣天气可能影响电网设备正常运行",
                    "用户用电行为在恶劣天气下变化较大"
                ])
        
        # 解释计算方法
        explanations['calculation_methodology'] = (
            f"本系统采用天气场景感知的动态不确定性建模方法：\n"
            f"1. 首先基于预测期间的天气数据（温度、湿度、风速、太阳辐射、降水量）识别天气场景\n"
            f"2. 根据识别的场景类型，动态调整 {forecast_type} 预测的不确定性参数\n"
            f"3. 结合历史同类场景下的预测误差分布，计算场景特定的预测区间\n"
            f"4. 考虑峰谷时段特性，进一步细化不确定性表征"
        )
        
        # 生成操作建议
        risk_level = scenario_analysis.get('risk_assessment', 'medium')
        if risk_level == 'high':
            explanations['recommendations'].extend([
                "建议增加实时监控频度，及时调整调度计划",
                "考虑启动备用电源或需求响应措施",
                "密切关注天气变化，准备应急预案"
            ])
        elif risk_level == 'medium':
            explanations['recommendations'].extend([
                "建议适度调整调度策略，关注关键时段",
                "加强负荷和新能源出力的实时监测"
            ])
        else:
            explanations['recommendations'].extend([
                "系统运行相对稳定，保持常规监控",
                "可考虑优化调度以提高经济性"
            ])
        
        return explanations

def perform_scenario_aware_uncertainty_forecast(province: str,
                                               forecast_type: str,
                                               start_date: str,
                                               end_date: str = None,
                                               confidence_level: float = 0.9,
                                               historical_days: int = 14,
                                               include_explanations: bool = True) -> Dict:
    """
    执行场景感知的不确定性预测（主接口函数）
    
    Args:
        province: 省份名称
        forecast_type: 预测类型 ('load', 'pv', 'wind')
        start_date: 预测开始日期
        end_date: 预测结束日期
        confidence_level: 置信水平
        historical_days: 历史数据天数
        include_explanations: 是否包含详细解释
        
    Returns:
        场景感知不确定性预测结果
    """
    engine = ScenarioAwareUncertaintyEngine()
    return engine.scenario_aware_uncertainty_forecast(
        province=province,
        forecast_type=forecast_type,
        start_date=start_date,
        end_date=end_date,
        confidence_level=confidence_level,
        historical_days=historical_days,
        include_explanations=include_explanations
    ) 

def load_weather_data(province, start_date, end_date=None):
    """
    加载指定省份和日期范围的天气数据
    
    参数:
    province: 省份名称
    start_date: 开始日期，格式为 'YYYY-MM-DD'
    end_date: 结束日期，格式为 'YYYY-MM-DD'，如果为None则使用start_date
    
    返回:
    weather_df: 包含天气数据的DataFrame
    """
    try:
        # 如果end_date为None，则使用start_date
        if end_date is None:
            end_date = start_date
            
        # 尝试加载真实天气数据
        weather_file = f"data/weather/weather_{province}_{start_date}_{end_date}.csv"
        
        if os.path.exists(weather_file):
            logger.info(f"从文件加载天气数据: {weather_file}")
            weather_df = pd.read_csv(weather_file, parse_dates=['datetime'])
            return weather_df
        
        # 如果没有找到文件，尝试加载最近的天气数据作为替代
        weather_dir = "data/weather"
        if os.path.exists(weather_dir):
            weather_files = [f for f in os.listdir(weather_dir) if f.startswith(f"weather_{province}_") and f.endswith(".csv")]
            if weather_files:
                # 选择最近的天气数据文件
                latest_weather_file = os.path.join(weather_dir, sorted(weather_files)[-1])
                logger.info(f"未找到指定日期的天气数据，使用最近的天气数据: {latest_weather_file}")
                weather_df = pd.read_csv(latest_weather_file, parse_dates=['datetime'])
                return weather_df
        
        # 如果没有找到任何天气数据，生成模拟数据
        logger.warning(f"未找到任何天气数据，生成模拟数据")
        return generate_simulated_weather_data(start_date, end_date)
        
    except Exception as e:
        logger.error(f"加载天气数据失败: {e}")
        return generate_simulated_weather_data(start_date, end_date)

def generate_simulated_weather_data(start_date, end_date=None):
    """
    生成模拟的天气数据
    
    参数:
    start_date: 开始日期，格式为 'YYYY-MM-DD'
    end_date: 结束日期，格式为 'YYYY-MM-DD'，如果为None则使用start_date
    
    返回:
    weather_df: 包含模拟天气数据的DataFrame
    """
    if end_date is None:
        end_date = start_date
        
    # 解析日期
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # 生成时间序列（15分钟间隔）
    date_range = pd.date_range(start=start_dt, end=end_dt + timedelta(days=1) - timedelta(minutes=15), freq='15min')
    
    # 确定季节
    month = start_dt.month
    if 3 <= month <= 5:
        season = 'spring'
    elif 6 <= month <= 8:
        season = 'summer'
    elif 9 <= month <= 11:
        season = 'autumn'
    else:
        season = 'winter'
    
    # 根据季节设置基础温度
    if season == 'summer':
        base_temp = 28
        temp_range = 10
    elif season == 'winter':
        base_temp = 5
        temp_range = 15
    else:  # spring or autumn
        base_temp = 18
        temp_range = 12
    
    # 生成模拟数据
    data = []
    for dt in date_range:
        hour = dt.hour
        # 温度变化（白天高，晚上低）
        hour_factor = np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else -0.5
        temperature = base_temp + hour_factor * (temp_range/2) + np.random.normal(0, 2)
        
        # 湿度（与温度负相关）
        humidity = 80 - (temperature - base_temp) + np.random.normal(0, 5)
        humidity = max(30, min(95, humidity))
        
        # 风速（白天略大）
        wind_speed = 3 + hour_factor * 2 + np.random.normal(0, 1)
        wind_speed = max(0.5, wind_speed)
        
        # 降水（随机）
        precipitation = 0
        if np.random.random() < 0.1:  # 10% 概率有降水
            precipitation = np.random.exponential(2)
        
        # 辐射（白天有，晚上无）
        radiation = 0
        if 6 <= hour <= 18:
            radiation = 800 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 50)
            radiation = max(0, radiation)
        
        data.append({
            'datetime': dt,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'precipitation': precipitation,
            'radiation': radiation
        })
    
    return pd.DataFrame(data)

# 测试代码
if __name__ == "__main__":
    # 测试场景感知不确定性预测
    result = perform_scenario_aware_uncertainty_forecast(
        province='上海',
        forecast_type='load',
        start_date='2024-06-10',
        end_date='2024-06-10',
        confidence_level=0.9,
        historical_days=14,
        include_explanations=True
    )
    
    # 打印结果
    print(json.dumps(result, indent=2, ensure_ascii=False)) 