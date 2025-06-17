#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的场景库 - 基于实际天气分析结果扩展场景定义
包含更完整的极端天气、典型场景和普通天气场景
"""

import numpy as np
import pandas as pd
import logging
from utils.weather_scenario_classifier import WeatherScenarioClassifier

logger = logging.getLogger(__name__)

class EnhancedScenarioLibrary(WeatherScenarioClassifier):
    """增强的场景库，包含更完整的场景定义"""
    
    def __init__(self):
        """初始化增强场景库"""
        super().__init__()
        
        # 扩展的场景库，基于实际数据分析结果
        self.scenarios.update({
            # === 极端天气场景 ===
            'extreme_storm_rain': {
                'name': '极端暴雨',
                'description': '强降水量超过30mm，湿度极高，风力较大，系统运行极不稳定',
                'uncertainty_multiplier': 3.5,
                'risk_level': 'very_high',
                'typical_features': {
                    'temperature': '10-25°C',
                    'humidity': '>90%',
                    'wind_speed': '>5m/s',
                    'precipitation': '>30mm',
                    'radiation': '极弱'
                }
            },
            
            'extreme_hot_humid': {
                'name': '极端高温高湿',
                'description': '高温伴随高湿度，空调负荷极高，电网压力巨大',
                'uncertainty_multiplier': 3.0,
                'risk_level': 'very_high',
                'typical_features': {
                    'temperature': '>32°C',
                    'humidity': '>80%',
                    'wind_speed': '2-6m/s',
                    'precipitation': '<10mm',
                    'radiation': '强'
                }
            },
            
            'extreme_strong_wind': {
                'name': '极端大风',
                'description': '风速超过10m/s，风电出力波动极大，系统稳定性受影响',
                'uncertainty_multiplier': 2.8,
                'risk_level': 'high',
                'typical_features': {
                    'temperature': '20-35°C',
                    'humidity': '60-85%',
                    'wind_speed': '>10m/s',
                    'precipitation': '<5mm',
                    'radiation': '中等'
                }
            },
            
            'extreme_heavy_rain': {
                'name': '特大暴雨',
                'description': '降水量超过50mm，伴随雷电，负荷模式异常，预测困难',
                'uncertainty_multiplier': 4.0,
                'risk_level': 'extreme',
                'typical_features': {
                    'temperature': '15-28°C',
                    'humidity': '>95%',
                    'wind_speed': '3-8m/s',
                    'precipitation': '>50mm',
                    'radiation': '极弱'
                }
            },
            
            # === 典型场景（基于实际分析） ===
            'typical_general_normal': {
                'name': '一般正常场景',
                'description': '温度适中，湿度正常，负荷稳定，对应实际分析的"一般场景0"',
                'uncertainty_multiplier': 1.0,
                'risk_level': 'low',
                'typical_features': {
                    'temperature': '5-12°C',
                    'humidity': '65-75%',
                    'wind_speed': '3-4m/s',
                    'precipitation': '<10mm',
                    'radiation': '中等'
                }
            },
            
            'typical_rainy_low_load': {
                'name': '多雨低负荷',
                'description': '降水较多，负荷相对较低，对应实际分析的"多雨低负荷"',
                'uncertainty_multiplier': 1.3,
                'risk_level': 'medium',
                'typical_features': {
                    'temperature': '18-25°C',
                    'humidity': '>75%',
                    'wind_speed': '3-4m/s',
                    'precipitation': '20-35mm',
                    'radiation': '弱'
                }
            },
            
            'typical_mild_humid_high_load': {
                'name': '温和高湿高负荷',
                'description': '温度适宜但湿度较高，负荷峰值明显，对应实际分析的"温和高湿高负荷"',
                'uncertainty_multiplier': 1.4,
                'risk_level': 'medium',
                'typical_features': {
                    'temperature': '28-33°C',
                    'humidity': '75-80%',
                    'wind_speed': '3-4m/s',
                    'precipitation': '<10mm',
                    'radiation': '中等到强'
                }
            },
            
            # === 普通天气场景变种 ===
            'normal_spring_mild': {
                'name': '春季温和',
                'description': '春季典型天气，温度回升，负荷平稳',
                'uncertainty_multiplier': 0.9,
                'risk_level': 'low',
                'typical_features': {
                    'temperature': '15-22°C',
                    'humidity': '60-70%',
                    'wind_speed': '2-5m/s',
                    'precipitation': '<5mm',
                    'radiation': '中等'
                }
            },
            
            'normal_summer_comfortable': {
                'name': '夏季舒适',
                'description': '夏季但不炎热，空调负荷适中',
                'uncertainty_multiplier': 1.1,
                'risk_level': 'low',
                'typical_features': {
                    'temperature': '25-30°C',
                    'humidity': '65-75%',
                    'wind_speed': '2-4m/s',
                    'precipitation': '<3mm',
                    'radiation': '强'
                }
            },
            
            'normal_autumn_stable': {
                'name': '秋季平稳',
                'description': '秋季凉爽，负荷和新能源出力稳定',
                'uncertainty_multiplier': 0.8,
                'risk_level': 'low',
                'typical_features': {
                    'temperature': '18-25°C',
                    'humidity': '55-65%',
                    'wind_speed': '2-4m/s',
                    'precipitation': '<2mm',
                    'radiation': '中等'
                }
            },
            
            'normal_winter_mild': {
                'name': '冬季温和',
                'description': '冬季但不严寒，采暖负荷适中',
                'uncertainty_multiplier': 1.2,
                'risk_level': 'low',
                'typical_features': {
                    'temperature': '8-15°C',
                    'humidity': '60-70%',
                    'wind_speed': '3-5m/s',
                    'precipitation': '<5mm',
                    'radiation': '弱到中等'
                }
            }
        })

    def _calculate_scenario_match_score(self, features, scenario_id):
        """增强的场景匹配评分算法"""
        weights = {
            'temperature': 0.25,
            'humidity': 0.20,
            'wind_speed': 0.20,
            'precipitation': 0.25,  # 提高降水权重
            'radiation': 0.10
        }
        
        # 获取基础评分
        if scenario_id in ['extreme_hot', 'extreme_cold', 'high_wind_sunny', 
                          'calm_cloudy', 'moderate_normal', 'storm_rain']:
            # 使用父类方法计算原有场景
            return super()._calculate_scenario_match_score(features, scenario_id)
        
        # 新增场景的评分逻辑
        temp = features.get('temperature', 20)
        humidity = features.get('humidity', 60)
        wind_speed = features.get('wind_speed', 3)
        precipitation = features.get('precipitation', 0)
        radiation = features.get('radiation', 500)
        
        score = 0
        
        # === 极端天气场景评分 ===
        if scenario_id == 'extreme_storm_rain':
            temp_score = self._score_range(temp, 10, 25, optimal_range=(15, 20))
            humidity_score = min(1.0, max(0, (humidity - 85) / 15))
            wind_score = min(1.0, max(0, (wind_speed - 4) / 6))
            precip_score = min(1.0, max(0, (precipitation - 25) / 25))
            radiation_score = max(0, 1 - radiation / 400)
            
        elif scenario_id == 'extreme_hot_humid':
            temp_score = min(1.0, max(0, (temp - 30) / 8))
            humidity_score = min(1.0, max(0, (humidity - 75) / 25))
            wind_score = self._score_range(wind_speed, 2, 6, optimal_range=(3, 5))
            precip_score = max(0, 1 - precipitation / 15)
            radiation_score = min(1.0, radiation / 800)
            
        elif scenario_id == 'extreme_strong_wind':
            temp_score = self._score_range(temp, 20, 35, optimal_range=(25, 30))
            humidity_score = self._score_range(humidity, 60, 85, optimal_range=(70, 80))
            wind_score = min(1.0, max(0, (wind_speed - 8) / 7))
            precip_score = max(0, 1 - precipitation / 8)
            radiation_score = self._score_range(radiation, 300, 700, optimal_range=(400, 600))
            
        elif scenario_id == 'extreme_heavy_rain':
            temp_score = self._score_range(temp, 15, 28, optimal_range=(20, 25))
            humidity_score = min(1.0, max(0, (humidity - 90) / 10))
            wind_score = self._score_range(wind_speed, 3, 8, optimal_range=(4, 6))
            precip_score = min(1.0, max(0, (precipitation - 40) / 40))
            radiation_score = max(0, 1 - radiation / 300)
            
        # === 典型场景评分 ===
        elif scenario_id == 'typical_general_normal':
            temp_score = self._score_range(temp, 5, 12, optimal_range=(7, 10))
            humidity_score = self._score_range(humidity, 65, 75, optimal_range=(68, 73))
            wind_score = self._score_range(wind_speed, 3, 4, optimal_range=(3.2, 3.8))
            precip_score = max(0, 1 - precipitation / 12)
            radiation_score = self._score_range(radiation, 300, 600, optimal_range=(400, 500))
            
        elif scenario_id == 'typical_rainy_low_load':
            temp_score = self._score_range(temp, 18, 25, optimal_range=(20, 23))
            humidity_score = min(1.0, max(0, (humidity - 70) / 30))
            wind_score = self._score_range(wind_speed, 3, 4, optimal_range=(3.5, 4))
            precip_score = self._score_range(precipitation, 20, 35, optimal_range=(25, 30))
            radiation_score = max(0, 1 - radiation / 500)
            
        elif scenario_id == 'typical_mild_humid_high_load':
            temp_score = self._score_range(temp, 28, 33, optimal_range=(29, 32))
            humidity_score = self._score_range(humidity, 75, 80, optimal_range=(76, 79))
            wind_score = self._score_range(wind_speed, 3, 4, optimal_range=(3.5, 4))
            precip_score = max(0, 1 - precipitation / 12)
            radiation_score = min(1.0, radiation / 700)
            
        # === 普通天气场景评分 ===
        elif scenario_id == 'normal_spring_mild':
            temp_score = self._score_range(temp, 15, 22, optimal_range=(18, 20))
            humidity_score = self._score_range(humidity, 60, 70, optimal_range=(63, 67))
            wind_score = self._score_range(wind_speed, 2, 5, optimal_range=(3, 4))
            precip_score = max(0, 1 - precipitation / 8)
            radiation_score = self._score_range(radiation, 400, 700, optimal_range=(500, 600))
            
        elif scenario_id == 'normal_summer_comfortable':
            temp_score = self._score_range(temp, 25, 30, optimal_range=(26, 29))
            humidity_score = self._score_range(humidity, 65, 75, optimal_range=(68, 72))
            wind_score = self._score_range(wind_speed, 2, 4, optimal_range=(2.5, 3.5))
            precip_score = max(0, 1 - precipitation / 5)
            radiation_score = min(1.0, radiation / 800)
            
        elif scenario_id == 'normal_autumn_stable':
            temp_score = self._score_range(temp, 18, 25, optimal_range=(20, 23))
            humidity_score = self._score_range(humidity, 55, 65, optimal_range=(58, 62))
            wind_score = self._score_range(wind_speed, 2, 4, optimal_range=(2.5, 3.5))
            precip_score = max(0, 1 - precipitation / 3)
            radiation_score = self._score_range(radiation, 350, 650, optimal_range=(450, 550))
            
        elif scenario_id == 'normal_winter_mild':
            temp_score = self._score_range(temp, 8, 15, optimal_range=(10, 13))
            humidity_score = self._score_range(humidity, 60, 70, optimal_range=(63, 67))
            wind_score = self._score_range(wind_speed, 3, 5, optimal_range=(3.5, 4.5))
            precip_score = max(0, 1 - precipitation / 8)
            radiation_score = self._score_range(radiation, 200, 500, optimal_range=(300, 400))
            
        else:
            # 未知场景，返回低分
            return 0.1
        
        # 计算加权总分
        score = (temp_score * weights['temperature'] + 
                humidity_score * weights['humidity'] + 
                wind_score * weights['wind_speed'] + 
                precip_score * weights['precipitation'] + 
                radiation_score * weights['radiation'])
        
        return max(0, min(1, score))  # 确保在0-1范围内

    def _score_range(self, value, min_val, max_val, optimal_range=None):
        """计算数值在指定范围内的得分"""
        if optimal_range:
            opt_min, opt_max = optimal_range
            if opt_min <= value <= opt_max:
                return 1.0
            elif value < opt_min:
                if value >= min_val:
                    return 1.0 - (opt_min - value) / (opt_min - min_val)
                else:
                    return 0.0
            else:  # value > opt_max
                if value <= max_val:
                    return 1.0 - (value - opt_max) / (max_val - opt_max)
                else:
                    return 0.0
        else:
            # 没有最优范围，线性评分
            if min_val <= value <= max_val:
                center = (min_val + max_val) / 2
                return 1.0 - abs(value - center) / (max_val - min_val)
            else:
                return 0.0

def create_enhanced_scenario_classifier():
    """创建增强的场景分类器实例"""
    return EnhancedScenarioLibrary() 