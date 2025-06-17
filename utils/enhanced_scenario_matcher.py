#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的场景匹配器 - 集成新能源大发判断的场景匹配
在原有场景匹配基础上，增加新能源出力状态的场景类型
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from utils.scenario_matcher import ScenarioMatcher

logger = logging.getLogger(__name__)

class EnhancedScenarioMatcher(ScenarioMatcher):
    """增强的场景匹配器，支持新能源大发场景识别"""
    
    def __init__(self, province=None):
        """
        初始化增强场景匹配器
        
        Args:
            province: 省份名称
        """
        super().__init__(province)
        
        # 扩展特征权重，增加新能源相关特征
        self.feature_weights = {
            # 天气特征权重
            'temperature_mean': 0.20,
            'humidity_mean': 0.10,
            'wind_speed_mean': 0.10,
            'precipitation_sum': 0.08,
            'solar_radiation_mean': 0.12,
            # 负荷特征权重  
            'load_mean': 0.20,
            'load_volatility': 0.10,
            # 新能源特征权重
            'pv_output_level': 0.05,
            'wind_output_level': 0.05
        }
        
        # 扩展特征标准化参数
        self.feature_ranges.update({
            'solar_radiation_mean': {'min': 0, 'max': 1000},
            'pv_output_level': {'min': 0, 'max': 1000},  # MW
            'wind_output_level': {'min': 0, 'max': 800}   # MW
        })
        
        # 增加新能源相关的典型场景
        self._create_enhanced_scenarios()
    
    def _create_enhanced_scenarios(self):
        """创建增强的典型场景（包含新能源大发场景）"""
        enhanced_scenarios = {
            "光伏大发低负荷": {
                "characteristics": {
                    "temperature_mean": 22.0,
                    "humidity_mean": 45.0,
                    "wind_speed_mean": 3.0,
                    "precipitation_sum": 0.0,
                    "solar_radiation_mean": 800.0,
                    "load_mean": 20000.0,
                    "load_volatility": 0.08,
                    "pv_output_level": 600.0,
                    "wind_output_level": 150.0
                },
                "percentage": 12.0,
                "description": "光照充足导致光伏大发，同时负荷较低，系统需要消纳大量光伏电力",
                "renewable_scenario_type": "pv_high"
            },
            "风电大发适中负荷": {
                "characteristics": {
                    "temperature_mean": 15.0,
                    "humidity_mean": 65.0,
                    "wind_speed_mean": 9.0,
                    "precipitation_sum": 2.0,
                    "solar_radiation_mean": 400.0,
                    "load_mean": 28000.0,
                    "load_volatility": 0.12,
                    "pv_output_level": 200.0,
                    "wind_output_level": 500.0
                },
                "percentage": 10.0,
                "description": "风速适中且持续，风电出力高，负荷适中，系统风电渗透率较高",
                "renewable_scenario_type": "wind_high"
            },
            "新能源双高中负荷": {
                "characteristics": {
                    "temperature_mean": 20.0,
                    "humidity_mean": 50.0,
                    "wind_speed_mean": 8.0,
                    "precipitation_sum": 0.0,
                    "solar_radiation_mean": 750.0,
                    "load_mean": 26000.0,
                    "load_volatility": 0.15,
                    "pv_output_level": 550.0,
                    "wind_output_level": 450.0
                },
                "percentage": 8.0,
                "description": "光伏和风电同时大发，负荷适中，系统新能源渗透率极高",
                "renewable_scenario_type": "dual_high"
            },
            "新能源低发高负荷": {
                "characteristics": {
                    "temperature_mean": 35.0,
                    "humidity_mean": 75.0,
                    "wind_speed_mean": 2.0,
                    "precipitation_sum": 0.0,
                    "solar_radiation_mean": 600.0,
                    "load_mean": 38000.0,
                    "load_volatility": 0.20,
                    "pv_output_level": 300.0,
                    "wind_output_level": 80.0
                },
                "percentage": 8.0,
                "description": "高温导致负荷激增，但新能源出力不足，系统供电压力大",
                "renewable_scenario_type": "renewable_low_load_high"
            }
        }
        
        # 将新场景添加到现有典型场景中
        if not hasattr(self, 'typical_scenarios') or not self.typical_scenarios:
            self.typical_scenarios = {}
        
        self.typical_scenarios.update(enhanced_scenarios)
        logger.info(f"增强场景匹配器已添加 {len(enhanced_scenarios)} 个新能源相关场景")
    
    def match_enhanced_scenario(self, current_features, renewable_predictions=None, province=None):
        """
        执行增强的场景匹配，考虑新能源出力状态
        
        Args:
            current_features: 当前的天气和负荷特征字典
            renewable_predictions: 新能源预测数据（可选）
            province: 省份名称（可选）
            
        Returns:
            dict: 包含最匹配场景信息和新能源分析的结果
        """
        if province and province != self.province:
            self.province = province
            self.load_typical_scenarios(province)
        
        # 扩展当前特征，加入新能源相关特征
        enhanced_features = self._extract_enhanced_features(current_features, renewable_predictions)
        
        # 执行基础场景匹配
        base_match_result = self.match_scenario(enhanced_features, province)
        
        if not base_match_result:
            return None
        
        # 分析新能源状态
        renewable_analysis = self._analyze_renewable_scenario_type(renewable_predictions)
        
        # 增强场景匹配结果
        enhanced_result = {
            **base_match_result,
            'renewable_analysis': renewable_analysis,
            'enhanced_scenario_classification': self._classify_enhanced_scenario(
                base_match_result, renewable_analysis
            ),
            'system_impact_assessment': self._assess_system_impact(
                base_match_result, renewable_analysis
            ),
            'forecast_complexity': self._assess_forecast_complexity(
                base_match_result, renewable_analysis
            )
        }
        
        return enhanced_result
    
    def _extract_enhanced_features(self, current_features, renewable_predictions):
        """提取增强特征，包含新能源相关信息"""
        enhanced_features = current_features.copy()
        
        # 添加太阳辐射特征（如果没有的话）
        if 'solar_radiation_mean' not in enhanced_features:
            # 根据温度和湿度估算太阳辐射
            temp = enhanced_features.get('temperature_mean', 20)
            humidity = enhanced_features.get('humidity_mean', 50)
            precip = enhanced_features.get('precipitation_sum', 0)
            
            # 简单的太阳辐射估算公式
            if precip > 10:
                estimated_radiation = 200
            elif temp > 25 and humidity < 60:
                estimated_radiation = 700
            elif temp > 20:
                estimated_radiation = 500
            else:
                estimated_radiation = 300
                
            enhanced_features['solar_radiation_mean'] = estimated_radiation
        
        # 从新能源预测中提取特征
        if renewable_predictions:
            # 光伏出力水平
            if 'pv' in renewable_predictions and renewable_predictions['pv'].get('predictions'):
                pv_predictions = renewable_predictions['pv']['predictions']
                enhanced_features['pv_output_level'] = np.mean(pv_predictions) if pv_predictions else 0
            else:
                enhanced_features['pv_output_level'] = 0
            
            # 风电出力水平
            if 'wind' in renewable_predictions and renewable_predictions['wind'].get('predictions'):
                wind_predictions = renewable_predictions['wind']['predictions']
                enhanced_features['wind_output_level'] = np.mean(wind_predictions) if wind_predictions else 0
            else:
                enhanced_features['wind_output_level'] = 0
        else:
            # 如果没有新能源预测，设置默认值
            enhanced_features['pv_output_level'] = 0
            enhanced_features['wind_output_level'] = 0
        
        return enhanced_features
    
    def _analyze_renewable_scenario_type(self, renewable_predictions):
        """分析新能源场景类型"""
        analysis = {
            'pv_status': 'unknown',
            'wind_status': 'unknown',
            'combined_status': 'unknown',
            'scenario_type': 'normal',
            'high_output_confidence': 0.0
        }
        
        if not renewable_predictions:
            return analysis
        
        # 分析光伏状态
        pv_high_periods = len(renewable_predictions.get('pv', {}).get('high_output_periods', []))
        if pv_high_periods >= 3:
            analysis['pv_status'] = 'high'
        elif pv_high_periods >= 1:
            analysis['pv_status'] = 'medium'
        else:
            analysis['pv_status'] = 'low'
        
        # 分析风电状态
        wind_high_periods = len(renewable_predictions.get('wind', {}).get('high_output_periods', []))
        if wind_high_periods >= 3:
            analysis['wind_status'] = 'high'
        elif wind_high_periods >= 1:
            analysis['wind_status'] = 'medium'
        else:
            analysis['wind_status'] = 'low'
        
        # 确定综合场景类型
        if analysis['pv_status'] == 'high' and analysis['wind_status'] == 'high':
            analysis['scenario_type'] = 'dual_high'
            analysis['high_output_confidence'] = 0.9
        elif analysis['pv_status'] == 'high':
            analysis['scenario_type'] = 'pv_high'
            analysis['high_output_confidence'] = 0.8
        elif analysis['wind_status'] == 'high':
            analysis['scenario_type'] = 'wind_high'
            analysis['high_output_confidence'] = 0.8
        elif analysis['pv_status'] == 'low' and analysis['wind_status'] == 'low':
            analysis['scenario_type'] = 'renewable_low'
            analysis['high_output_confidence'] = 0.1
        else:
            analysis['scenario_type'] = 'normal'
            analysis['high_output_confidence'] = 0.5
        
        return analysis
    
    def _classify_enhanced_scenario(self, base_match_result, renewable_analysis):
        """对增强场景进行分类"""
        matched_scenario = base_match_result.get('matched_scenario', {})
        scenario_name = matched_scenario.get('name', '')
        renewable_type = renewable_analysis.get('scenario_type', 'normal')
        
        # 构建增强场景分类
        classification = {
            'primary_category': 'unknown',
            'renewable_category': renewable_type,
            'complexity_level': 'medium',
            'system_stress_level': 'medium'
        }
        
        # 基于场景名称和新能源状态进行分类
        if renewable_type == 'dual_high':
            classification['primary_category'] = '新能源双高'
            classification['complexity_level'] = 'high'
            classification['system_stress_level'] = 'high'
        elif renewable_type == 'pv_high':
            classification['primary_category'] = '光伏大发'
            classification['complexity_level'] = 'medium'
            classification['system_stress_level'] = 'medium'
        elif renewable_type == 'wind_high':
            classification['primary_category'] = '风电大发'
            classification['complexity_level'] = 'medium'
            classification['system_stress_level'] = 'medium'
        elif renewable_type == 'renewable_low':
            if '高温' in scenario_name or '低温' in scenario_name:
                classification['primary_category'] = '新能源低发+极端负荷'
                classification['complexity_level'] = 'very_high'
                classification['system_stress_level'] = 'very_high'
            else:
                classification['primary_category'] = '新能源低发'
                classification['complexity_level'] = 'high'
                classification['system_stress_level'] = 'high'
        else:
            classification['primary_category'] = scenario_name
        
        return classification
    
    def _assess_system_impact(self, base_match_result, renewable_analysis):
        """评估系统影响"""
        renewable_type = renewable_analysis.get('scenario_type', 'normal')
        confidence = renewable_analysis.get('high_output_confidence', 0.5)
        
        impact_assessment = {
            'flexibility_requirement': 'medium',
            'reserve_requirement': 'normal',
            'grid_stability_risk': 'low',
            'economic_impact': 'neutral',
            'operational_complexity': 'medium'
        }
        
        if renewable_type == 'dual_high':
            impact_assessment.update({
                'flexibility_requirement': 'very_high',
                'reserve_requirement': 'high',
                'grid_stability_risk': 'medium',
                'economic_impact': 'cost_reduction_but_complexity',
                'operational_complexity': 'very_high'
            })
        elif renewable_type in ['pv_high', 'wind_high']:
            impact_assessment.update({
                'flexibility_requirement': 'high',
                'reserve_requirement': 'medium',
                'grid_stability_risk': 'low',
                'economic_impact': 'cost_reduction',
                'operational_complexity': 'high'
            })
        elif renewable_type == 'renewable_low':
            impact_assessment.update({
                'flexibility_requirement': 'low',
                'reserve_requirement': 'high',
                'grid_stability_risk': 'medium',
                'economic_impact': 'cost_increase',
                'operational_complexity': 'medium'
            })
        
        return impact_assessment
    
    def _assess_forecast_complexity(self, base_match_result, renewable_analysis):
        """评估预测复杂度"""
        renewable_type = renewable_analysis.get('scenario_type', 'normal')
        base_difficulty = base_match_result.get('matched_scenario', {}).get('description', '')
        
        complexity_assessment = {
            'load_forecast_difficulty': 'medium',
            'renewable_forecast_difficulty': 'medium',
            'overall_uncertainty': 'medium',
            'recommended_forecast_frequency': 'normal',
            'key_monitoring_points': []
        }
        
        # 基于新能源场景类型调整复杂度
        if renewable_type == 'dual_high':
            complexity_assessment.update({
                'load_forecast_difficulty': 'high',
                'renewable_forecast_difficulty': 'very_high',
                'overall_uncertainty': 'high',
                'recommended_forecast_frequency': 'increased',
                'key_monitoring_points': ['光伏出力波动', '风电出力波动', '净负荷变化', '系统调节能力']
            })
        elif renewable_type in ['pv_high', 'wind_high']:
            complexity_assessment.update({
                'load_forecast_difficulty': 'medium',
                'renewable_forecast_difficulty': 'high',
                'overall_uncertainty': 'medium-high',
                'recommended_forecast_frequency': 'normal',
                'key_monitoring_points': [f'{renewable_type.split("_")[0]}出力变化', '净负荷平衡']
            })
        elif renewable_type == 'renewable_low':
            complexity_assessment.update({
                'load_forecast_difficulty': 'high',
                'renewable_forecast_difficulty': 'medium',
                'overall_uncertainty': 'high',
                'recommended_forecast_frequency': 'normal',
                'key_monitoring_points': ['常规电源启停', '负荷峰值', '备用容量']
            })
        
        return complexity_assessment
    
    def generate_enhanced_scenario_summary(self, enhanced_match_result):
        """生成增强场景匹配的综合摘要"""
        if not enhanced_match_result:
            return "无法生成场景摘要"
        
        matched_scenario = enhanced_match_result.get('matched_scenario', {})
        renewable_analysis = enhanced_match_result.get('renewable_analysis', {})
        classification = enhanced_match_result.get('enhanced_scenario_classification', {})
        system_impact = enhanced_match_result.get('system_impact_assessment', {})
        
        summary = f"""
=== 增强场景匹配摘要 ===
匹配场景: {matched_scenario.get('name', '未知')}
相似度: {matched_scenario.get('similarity_percentage', 0):.1f}%
新能源状态: {renewable_analysis.get('scenario_type', '正常')}
系统复杂度: {classification.get('complexity_level', '中等')}
灵活性需求: {system_impact.get('flexibility_requirement', '中等')}
预测难度: {system_impact.get('operational_complexity', '中等')}

场景描述: {matched_scenario.get('description', '无描述')}
系统影响: {system_impact.get('economic_impact', '中性')}
关键监控点: {', '.join(enhanced_match_result.get('forecast_complexity', {}).get('key_monitoring_points', []))}
        """.strip()
        
        return summary

def create_enhanced_scenario_matcher(province=None):
    """创建增强的场景匹配器实例"""
    return EnhancedScenarioMatcher(province=province)

# 测试函数
if __name__ == "__main__":
    matcher = EnhancedScenarioMatcher('上海')
    
    # 测试特征
    test_features = {
        'temperature_mean': 22.0,
        'humidity_mean': 45.0,
        'wind_speed_mean': 3.0,
        'precipitation_sum': 0.0,
        'load_mean': 25000.0,
        'load_volatility': 0.12
    }
    
    # 测试新能源预测
    test_renewable = {
        'pv': {'high_output_periods': [{'duration_hours': 4}, {'duration_hours': 3}]},
        'wind': {'high_output_periods': [{'duration_hours': 2}]}
    }
    
    result = matcher.match_enhanced_scenario(test_features, test_renewable)
    summary = matcher.generate_enhanced_scenario_summary(result)
    
    print(summary) 