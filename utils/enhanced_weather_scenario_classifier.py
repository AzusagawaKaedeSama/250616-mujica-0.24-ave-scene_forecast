#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强的天气场景分类器
在原有天气场景基础上，增加新能源大发状态的综合判断
支持光伏、风电出力预测与天气场景的联合分析
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from utils.weather_scenario_classifier import WeatherScenarioClassifier

logger = logging.getLogger(__name__)

class EnhancedWeatherScenarioClassifier(WeatherScenarioClassifier):
    """增强的天气场景分类器，集成新能源大发判断"""
    
    def __init__(self):
        """初始化增强的天气场景分类器"""
        super().__init__()
        
        # 扩展场景定义，增加新能源相关场景
        self.renewable_scenarios = {
            'pv_high_output': {
                'name': '光伏大发',
                'description': '光照充足，光伏出力高，系统需要消纳大量光伏电力',
                'uncertainty_multiplier': 1.2,
                'risk_level': 'medium',
                'renewable_impact': 'high_pv'
            },
            'wind_high_output': {
                'name': '风电大发',
                'description': '风速适中且持续，风电出力高，系统风电渗透率高',
                'uncertainty_multiplier': 1.3,
                'risk_level': 'medium',
                'renewable_impact': 'high_wind'
            },
            'renewable_dual_high': {
                'name': '新能源双高',
                'description': '光伏和风电同时大发，系统新能源渗透率极高，调节压力大',
                'uncertainty_multiplier': 1.5,
                'risk_level': 'medium-high',
                'renewable_impact': 'dual_high'
            },
            'renewable_intermittent': {
                'name': '新能源间歇',
                'description': '天气快速变化，新能源出力间歇性强，预测难度大',
                'uncertainty_multiplier': 2.2,
                'risk_level': 'high',
                'renewable_impact': 'intermittent'
            },
            'renewable_low_extreme_weather': {
                'name': '新能源低发+极端天气',
                'description': '极端天气导致新能源出力低，负荷需求异常，系统压力巨大',
                'uncertainty_multiplier': 3.2,
                'risk_level': 'very_high',
                'renewable_impact': 'very_low'
            }
        }
        
        # 合并所有场景
        self.scenarios.update(self.renewable_scenarios)
    
    def identify_renewable_enhanced_scenario(self, weather_data, renewable_predictions=None):
        """
        识别增强的新能源天气场景
        
        参数:
        weather_data: 天气数据（DataFrame或dict）
        renewable_predictions: 新能源预测数据（可选）
        
        返回:
        enhanced_scenario_info: 增强的场景信息
        """
        # 首先进行基础天气场景识别
        base_scenario = self.identify_scenario(weather_data)
        
        # 分析新能源相关特征
        renewable_features = self._extract_renewable_features(weather_data)
        
        # 如果有新能源预测数据，结合预测进行分析
        if renewable_predictions:
            renewable_analysis = self._analyze_renewable_predictions(renewable_predictions)
            renewable_features.update(renewable_analysis)
        
        # 计算新能源场景得分
        renewable_scenario_scores = self._calculate_renewable_scenario_scores(renewable_features)
        
        # 综合确定最终场景
        enhanced_scenario = self._determine_enhanced_scenario(
            base_scenario, renewable_scenario_scores, renewable_features
        )
        
        return {
            'enhanced_scenario': enhanced_scenario,
            'base_weather_scenario': base_scenario,
            'renewable_features': renewable_features,
            'renewable_scenario_scores': renewable_scenario_scores
        }
    
    def _extract_renewable_features(self, weather_data):
        """提取新能源相关的天气特征"""
        if isinstance(weather_data, pd.DataFrame):
            features = {
                'temperature': weather_data.get('temperature', pd.Series([20])).mean(),
                'wind_speed': weather_data.get('wind_speed', pd.Series([3])).mean(),
                'radiation': weather_data.get('solar_radiation', weather_data.get('radiation', pd.Series([500]))).mean(),
                'precipitation': weather_data.get('precipitation', pd.Series([0])).sum()
            }
        else:
            features = weather_data.copy()
        
        # 计算新能源相关指标
        features['pv_favorability'] = self._calculate_pv_favorability(features)
        features['wind_favorability'] = self._calculate_wind_favorability(features)
        
        return features
    
    def _calculate_pv_favorability(self, features):
        """计算光伏发电适宜度"""
        radiation_score = min(1.0, features.get('radiation', 0) / 800)
        temp_score = 1.0 if 15 <= features.get('temperature', 20) <= 30 else 0.5
        precipitation_penalty = max(0, 1 - features.get('precipitation', 0) / 10)
        
        return (radiation_score * 0.6 + temp_score * 0.2 + precipitation_penalty * 0.2)
    
    def _calculate_wind_favorability(self, features):
        """计算风电发电适宜度"""
        wind_speed = features.get('wind_speed', 0)
        
        if 6 <= wind_speed <= 12:
            return 1.0
        elif 3 <= wind_speed < 6:
            return (wind_speed - 3) / 3
        elif 12 < wind_speed <= 15:
            return 1.0 - (wind_speed - 12) / 3
        else:
            return 0.1
    
    def _analyze_renewable_predictions(self, renewable_predictions):
        """分析新能源预测数据"""
        analysis = {
            'pv_high_periods': 0,
            'wind_high_periods': 0,
            'combined_high_periods': 0
        }
        
        if 'pv' in renewable_predictions:
            analysis['pv_high_periods'] = len(renewable_predictions['pv'].get('high_output_periods', []))
        
        if 'wind' in renewable_predictions:
            analysis['wind_high_periods'] = len(renewable_predictions['wind'].get('high_output_periods', []))
        
        combined_data = renewable_predictions.get('combined_high_output', {})
        analysis['combined_high_periods'] = combined_data.get('analysis', {}).get('both_high_periods', 0)
        
        return analysis
    
    def _calculate_renewable_scenario_scores(self, renewable_features):
        """计算新能源场景得分"""
        scores = {}
        
        # 光伏大发场景
        scores['pv_high_output'] = renewable_features.get('pv_favorability', 0)
        
        # 风电大发场景
        scores['wind_high_output'] = renewable_features.get('wind_favorability', 0)
        
        # 双重大发场景
        pv_score = renewable_features.get('pv_favorability', 0)
        wind_score = renewable_features.get('wind_favorability', 0)
        scores['renewable_dual_high'] = min(pv_score + wind_score, 1.0) if pv_score > 0.6 and wind_score > 0.6 else 0
        
        # 间歇性场景
        temp = renewable_features.get('temperature', 20)
        precip = renewable_features.get('precipitation', 0)
        if temp < 5 or temp > 35 or precip > 10:
            scores['renewable_intermittent'] = 0.8
        else:
            scores['renewable_intermittent'] = 0.2
        
        # 低发+极端天气场景
        if temp < 0 or temp > 40 or precip > 30:
            scores['renewable_low_extreme_weather'] = 1.0
        else:
            scores['renewable_low_extreme_weather'] = 0.1
        
        return scores
    
    def _determine_enhanced_scenario(self, base_scenario, renewable_scores, renewable_features):
        """确定最终的增强场景"""
        # 获取最高得分的新能源场景
        best_renewable_scenario = max(renewable_scores.items(), key=lambda x: x[1])
        best_renewable_name, best_renewable_score = best_renewable_scenario
        
        # 如果新能源场景得分足够高，则使用新能源场景
        if best_renewable_score > 0.7:
            selected_scenario = self.renewable_scenarios[best_renewable_name].copy()
            selected_scenario['match_score'] = best_renewable_score
            selected_scenario['scenario_type'] = 'renewable_enhanced'
        else:
            # 否则使用基础天气场景
            selected_scenario = base_scenario.copy()
            selected_scenario['scenario_type'] = 'weather_primary'
        
        return selected_scenario

def create_enhanced_weather_scenario_classifier():
    """创建增强的天气场景分类器实例"""
    return EnhancedWeatherScenarioClassifier()

# 测试函数
if __name__ == "__main__":
    classifier = EnhancedWeatherScenarioClassifier()
    
    # 测试数据
    test_weather = {
        'temperature': 25,
        'humidity': 45,
        'wind_speed': 8,
        'precipitation': 0,
        'radiation': 750
    }
    
    test_renewable = {
        'pv': {'high_output_periods': [{'duration_hours': 4}]},
        'wind': {'high_output_periods': [{'duration_hours': 3}]},
        'combined_high_output': {'analysis': {'both_high_periods': 1}}
    }
    
    result = classifier.identify_renewable_enhanced_scenario(test_weather, test_renewable)
    
    print("=== 增强场景分析结果 ===")
    print(f"场景名称: {result['enhanced_scenario']['name']}")
    print(f"场景描述: {result['enhanced_scenario']['description']}")
    print(f"风险等级: {result['enhanced_scenario']['risk_level']}")
    print(f"不确定性系数: {result['enhanced_scenario']['uncertainty_multiplier']}") 