#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景匹配器 - 基于欧式距离计算与典型场景的相似度
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ScenarioMatcher:
    """基于欧式距离的场景匹配器"""
    
    def __init__(self, province=None):
        """
        初始化场景匹配器
        
        Args:
            province: 省份名称，如果不指定则需要在调用时指定
        """
        self.province = province
        self.typical_scenarios = {}
        self.feature_weights = {
            # 天气特征权重
            'temperature_mean': 0.25,
            'humidity_mean': 0.15,
            'wind_speed_mean': 0.10,
            'precipitation_sum': 0.10,
            # 负荷特征权重  
            'load_mean': 0.25,
            'load_volatility': 0.15
        }
        
        # 特征标准化参数（用于归一化不同量纲的特征）
        self.feature_ranges = {
            'temperature_mean': {'min': -10, 'max': 40},
            'humidity_mean': {'min': 0, 'max': 100},
            'wind_speed_mean': {'min': 0, 'max': 15},
            'precipitation_sum': {'min': 0, 'max': 100},
            'load_mean': {'min': 10000, 'max': 50000},
            'load_volatility': {'min': 0, 'max': 0.5}
        }
        
        if self.province:
            self.load_typical_scenarios(self.province)
    
    def load_typical_scenarios(self, province):
        """加载指定省份的典型场景数据"""
        try:
            # 从场景分析结果文件加载
            results_file = f"results/weather_scenario_analysis/2024/{province}_analysis_results.json"
            
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.typical_scenarios = data.get('typical_days', {})
                    logger.info(f"成功加载 {province} 的 {len(self.typical_scenarios)} 个典型场景")
            else:
                logger.warning(f"典型场景文件不存在: {results_file}")
                # 使用默认场景
                self._create_default_scenarios()
                
        except Exception as e:
            logger.error(f"加载典型场景失败: {e}")
            self._create_default_scenarios()
    
    def _create_default_scenarios(self):
        """创建默认典型场景（当没有历史数据时使用）"""
        self.typical_scenarios = {
            "温和正常": {
                "characteristics": {
                    "temperature_mean": 20.0,
                    "humidity_mean": 60.0,
                    "wind_speed_mean": 3.0,
                    "precipitation_sum": 0.0,
                    "load_mean": 25000.0,
                    "load_volatility": 0.12
                },
                "percentage": 40.0,
                "description": "温和正常天气下的标准负荷模式"
            },
            "高温高负荷": {
                "characteristics": {
                    "temperature_mean": 32.0,
                    "humidity_mean": 70.0,
                    "wind_speed_mean": 2.5,
                    "precipitation_sum": 0.0,
                    "load_mean": 35000.0,
                    "load_volatility": 0.18
                },
                "percentage": 25.0,
                "description": "高温天气导致空调负荷增加"
            },
            "低温高负荷": {
                "characteristics": {
                    "temperature_mean": 5.0,
                    "humidity_mean": 55.0,
                    "wind_speed_mean": 4.0,
                    "precipitation_sum": 5.0,
                    "load_mean": 32000.0,
                    "load_volatility": 0.15
                },
                "percentage": 20.0,
                "description": "低温天气导致采暖负荷增加"
            },
            "雨天低负荷": {
                "characteristics": {
                    "temperature_mean": 18.0,
                    "humidity_mean": 85.0,
                    "wind_speed_mean": 3.5,
                    "precipitation_sum": 25.0,
                    "load_mean": 22000.0,
                    "load_volatility": 0.10
                },
                "percentage": 15.0,
                "description": "雨天通常负荷较低且相对稳定"
            }
        }
        logger.info("使用默认典型场景")
    
    def normalize_feature(self, feature_name, value):
        """归一化特征值到0-1范围"""
        if feature_name not in self.feature_ranges:
            return value
        
        min_val = self.feature_ranges[feature_name]['min']
        max_val = self.feature_ranges[feature_name]['max']
        
        # 线性归一化到0-1
        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(1, normalized))  # 确保在0-1范围内
    
    def calculate_euclidean_distance(self, features, scenario_characteristics):
        """计算当前特征与典型场景的欧式距离"""
        distance = 0.0
        used_features = []
        
        for feature_name, weight in self.feature_weights.items():
            if feature_name in features and feature_name in scenario_characteristics:
                # 归一化特征值
                current_val = self.normalize_feature(feature_name, features[feature_name])
                typical_val = self.normalize_feature(feature_name, scenario_characteristics[feature_name])
                
                # 计算加权欧式距离的平方
                diff = (current_val - typical_val) ** 2
                distance += weight * diff
                used_features.append(feature_name)
        
        # 返回欧式距离（开方）
        euclidean_distance = np.sqrt(distance)
        
        logger.debug(f"计算距离时使用的特征: {used_features}")
        logger.debug(f"欧式距离: {euclidean_distance:.4f}")
        
        return euclidean_distance
    
    def match_scenario(self, current_features, province=None):
        """
        匹配最相似的典型场景
        
        Args:
            current_features: 当前的天气和负荷特征字典
            province: 省份名称（可选，如果初始化时已指定则不需要）
            
        Returns:
            dict: 包含最匹配场景信息和所有场景相似度的结果
        """
        if province and province != self.province:
            self.province = province
            self.load_typical_scenarios(province)
        
        if not self.typical_scenarios:
            logger.error("没有可用的典型场景数据")
            return None
        
        logger.info(f"开始场景匹配，当前特征: {current_features}")
        
        scenario_distances = {}
        scenario_similarities = {}
        
        # 计算与每个典型场景的距离
        for scenario_name, scenario_data in self.typical_scenarios.items():
            characteristics = scenario_data.get('characteristics', {})
            
            if not characteristics:
                logger.warning(f"场景 {scenario_name} 缺少特征数据")
                continue
            
            # 计算欧式距离
            distance = self.calculate_euclidean_distance(current_features, characteristics)
            scenario_distances[scenario_name] = distance
            
            # 转换为相似度（距离越小，相似度越高）
            # 使用指数衰减函数：similarity = exp(-distance)
            similarity = np.exp(-distance * 2)  # 乘以2增加区分度
            scenario_similarities[scenario_name] = similarity
            
            logger.debug(f"场景 {scenario_name}: 距离={distance:.4f}, 相似度={similarity:.4f}")
        
        if not scenario_distances:
            logger.error("无法计算任何场景的距离")
            return None
        
        # 找到最相似的场景（距离最小）
        best_scenario_name = min(scenario_distances, key=scenario_distances.get)
        best_distance = scenario_distances[best_scenario_name]
        best_similarity = scenario_similarities[best_scenario_name]
        
        # 计算相似度排名
        sorted_scenarios = sorted(scenario_similarities.items(), 
                                key=lambda x: x[1], reverse=True)
        
        result = {
            'matched_scenario': {
                'name': best_scenario_name,
                'distance': best_distance,
                'similarity': best_similarity,
                'similarity_percentage': best_similarity * 100,
                'characteristics': self.typical_scenarios[best_scenario_name]['characteristics'],
                'description': self.typical_scenarios[best_scenario_name].get('description', ''),
                'typical_percentage': self.typical_scenarios[best_scenario_name].get('percentage', 0)
            },
            'all_scenarios': [
                {
                    'name': name,
                    'similarity': sim,
                    'similarity_percentage': sim * 100,
                    'distance': scenario_distances[name],
                    'rank': idx + 1
                }
                for idx, (name, sim) in enumerate(sorted_scenarios)
            ],
            'feature_analysis': self._analyze_feature_contributions(
                current_features, 
                self.typical_scenarios[best_scenario_name]['characteristics']
            ),
            'confidence_level': self._calculate_confidence(scenario_similarities)
        }
        
        logger.info(f"最佳匹配场景: {best_scenario_name} (相似度: {best_similarity:.2f})")
        
        return result
    
    def _analyze_feature_contributions(self, current_features, best_scenario_characteristics):
        """分析各特征对场景匹配的贡献度"""
        contributions = {}
        
        for feature_name, weight in self.feature_weights.items():
            if feature_name in current_features and feature_name in best_scenario_characteristics:
                current_val = self.normalize_feature(feature_name, current_features[feature_name])
                typical_val = self.normalize_feature(feature_name, best_scenario_characteristics[feature_name])
                
                # 特征差异
                diff = abs(current_val - typical_val)
                
                # 特征重要性（权重 * (1 - 差异)）
                importance = weight * (1 - diff)
                
                contributions[feature_name] = {
                    'current_value': current_features[feature_name],
                    'typical_value': best_scenario_characteristics[feature_name],
                    'normalized_diff': diff,
                    'weight': weight,
                    'contribution': importance
                }
        
        return contributions
    
    def _calculate_confidence(self, scenario_similarities):
        """计算匹配置信度"""
        similarities = list(scenario_similarities.values())
        
        if len(similarities) < 2:
            return 0.5
        
        # 排序相似度
        similarities.sort(reverse=True)
        
        # 最高相似度与第二高相似度的差值作为置信度指标
        confidence = similarities[0] - similarities[1] if len(similarities) > 1 else similarities[0]
        
        # 归一化到0-1范围
        return min(1.0, max(0.0, confidence))
    
    def create_scenario_summary(self, match_result):
        """创建场景匹配结果摘要"""
        if not match_result:
            return "无法进行场景匹配"
        
        matched = match_result['matched_scenario']
        confidence = match_result['confidence_level']
        
        # 生成置信度描述
        if confidence > 0.7:
            confidence_desc = "高"
        elif confidence > 0.4:
            confidence_desc = "中"
        else:
            confidence_desc = "低"
        
        summary = f"""
场景匹配结果:
- 最匹配场景: {matched['name']}
- 相似度: {matched['similarity_percentage']:.1f}%
- 置信度: {confidence_desc} ({confidence:.2f})
- 场景特点: {matched.get('description', '无描述')}
- 历史占比: {matched['typical_percentage']:.1f}%

前3名相似场景:
"""
        
        for scenario in match_result['all_scenarios'][:3]:
            summary += f"{scenario['rank']}. {scenario['name']}: {scenario['similarity_percentage']:.1f}%\n"
        
        return summary

def create_scenario_matcher(province=None):
    """创建场景匹配器实例的工厂函数"""
    return ScenarioMatcher(province=province)

if __name__ == "__main__":
    # 测试代码
    matcher = ScenarioMatcher('上海')
    
    # 模拟当前特征
    test_features = {
        'temperature_mean': 28.5,
        'humidity_mean': 75.0,
        'wind_speed_mean': 3.2,
        'precipitation_sum': 0.0,
        'load_mean': 30000.0,
        'load_volatility': 0.16
    }
    
    result = matcher.match_scenario(test_features)
    
    if result:
        print("=== 场景匹配测试结果 ===")
        print(matcher.create_scenario_summary(result))
        print("\n=== 详细结果 ===")
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2)) 