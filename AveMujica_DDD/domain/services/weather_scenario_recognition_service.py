from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math
from dataclasses import dataclass

from AveMujica_DDD.domain.aggregates.weather_scenario import (
    WeatherScenario, ScenarioType, PREDEFINED_SCENARIOS
)

@dataclass
class WeatherFeatures:
    """天气特征数据类"""
    temperature: float  # 温度 (°C)
    humidity: float     # 湿度 (%)
    wind_speed: float   # 风速 (m/s)
    precipitation: float  # 降水量 (mm)
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'temperature': self.temperature,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'precipitation': self.precipitation
        }

@dataclass
class ScenarioMatchResult:
    """场景匹配结果"""
    matched_scenario: WeatherScenario
    similarity_score: float  # 相似度分数 (0-1)
    confidence_level: str   # 置信度等级 ("高", "中", "低")
    distance: float        # 欧式距离
    feature_contributions: Dict[str, float]  # 各特征贡献度
    ranking: List[Tuple[WeatherScenario, float]]  # 前三名场景排名

class WeatherScenarioRecognitionService:
    """
    天气场景识别领域服务。
    实现智能的17种天气场景识别算法，支持欧式距离匹配和相似度计算。
    """
    
    def __init__(self):
        """初始化场景识别服务"""
        # 特征权重配置 (基于气象学重要性)
        self.feature_weights = {
            'temperature': 0.25,    # 温度对负荷影响最大
            'humidity': 0.15,       # 湿度影响舒适度需求
            'wind_speed': 0.10,     # 风速影响风电出力
            'precipitation': 0.10   # 降水影响系统稳定性
        }
        
        # 置信度阈值配置
        self.confidence_thresholds = {
            'high': 0.85,    # 高置信度：相似度>85%
            'medium': 0.65,  # 中等置信度：相似度65%-85%
            'low': 0.0       # 低置信度：相似度<65%
        }
        
        # 负荷特征权重 (如果提供负荷数据)
        self.load_feature_weights = {
            'average_load': 0.25,     # 平均负荷水平
            'load_volatility': 0.15   # 负荷波动率
        }

    def recognize_scenario(
        self, 
        weather_features: WeatherFeatures,
        load_features: Optional[Dict[str, float]] = None,
        use_enhanced_matching: bool = True
    ) -> ScenarioMatchResult:
        """
        识别天气场景。
        
        Args:
            weather_features: 天气特征数据
            load_features: 负荷特征数据 (可选)
            use_enhanced_matching: 是否使用增强匹配算法
            
        Returns:
            场景匹配结果
        """
        # 1. 基于规则的快速识别
        rule_based_scenario = self._rule_based_recognition(weather_features)
        
        # 2. 基于相似度的智能匹配
        similarity_results = self._similarity_based_matching(
            weather_features, load_features, use_enhanced_matching
        )
        
        # 3. 综合决策 (优先考虑相似度匹配结果)
        final_scenario, similarity_score, distance = similarity_results[0]
        
        # 4. 计算置信度
        confidence_level = self._calculate_confidence_level(similarity_score)
        
        # 5. 计算特征贡献度
        feature_contributions = self._calculate_feature_contributions(
            weather_features, final_scenario.typical_features
        )
        
        # 6. 生成排名 (前三名) - 只保留场景和相似度
        ranking = [(scenario, similarity) for scenario, similarity, _ in similarity_results[:3]]
        
        return ScenarioMatchResult(
            matched_scenario=final_scenario,
            similarity_score=similarity_score,
            confidence_level=confidence_level,
            distance=distance,
            feature_contributions=feature_contributions,
            ranking=ranking
        )

    def batch_recognize_scenarios(
        self, 
        weather_data_list: List[WeatherFeatures]
    ) -> List[ScenarioMatchResult]:
        """
        批量识别天气场景。
        
        Args:
            weather_data_list: 天气数据列表
            
        Returns:
            场景匹配结果列表
        """
        results = []
        for weather_data in weather_data_list:
            result = self.recognize_scenario(weather_data)
            results.append(result)
        
        return results

    def get_scenario_statistics(
        self, 
        recognition_results: List[ScenarioMatchResult]
    ) -> Dict[str, Any]:
        """
        计算场景识别统计信息。
        
        Args:
            recognition_results: 识别结果列表
            
        Returns:
            统计信息字典
        """
        if not recognition_results:
            return {}
        
        # 场景分布统计
        scenario_counts = {}
        confidence_counts = {"高": 0, "中": 0, "低": 0}
        total_similarity = 0
        
        for result in recognition_results:
            scenario_type = result.matched_scenario.scenario_type.value
            scenario_counts[scenario_type] = scenario_counts.get(scenario_type, 0) + 1
            confidence_counts[result.confidence_level] += 1
            total_similarity += result.similarity_score
        
        # 主导场景识别
        dominant_scenario = max(scenario_counts.items(), key=lambda x: x[1])
        
        return {
            'total_samples': len(recognition_results),
            'scenario_distribution': scenario_counts,
            'confidence_distribution': confidence_counts,
            'average_similarity': total_similarity / len(recognition_results),
            'dominant_scenario': {
                'type': dominant_scenario[0],
                'count': dominant_scenario[1],
                'percentage': (dominant_scenario[1] / len(recognition_results)) * 100
            }
        }

    def _rule_based_recognition(self, weather_features: WeatherFeatures) -> WeatherScenario:
        """
        基于规则的快速场景识别。
        """
        temp = weather_features.temperature
        humidity = weather_features.humidity
        wind_speed = weather_features.wind_speed
        precipitation = weather_features.precipitation
        
        # 极端天气场景识别规则
        if precipitation > 50 and humidity > 95:
            return PREDEFINED_SCENARIOS[ScenarioType.EXTREME_HEAVY_RAIN]
        elif precipitation > 30 and humidity > 90:
            return PREDEFINED_SCENARIOS[ScenarioType.EXTREME_STORM_RAIN]
        elif temp > 32 and humidity > 80:
            return PREDEFINED_SCENARIOS[ScenarioType.EXTREME_HOT_HUMID]
        elif wind_speed > 10:
            return PREDEFINED_SCENARIOS[ScenarioType.EXTREME_STRONG_WIND]
        elif temp > 35:
            return PREDEFINED_SCENARIOS[ScenarioType.EXTREME_HOT]
        elif temp < 0:
            return PREDEFINED_SCENARIOS[ScenarioType.EXTREME_COLD]
        
        # 普通天气场景识别
        elif precipitation > 20 and wind_speed > 7:
            return PREDEFINED_SCENARIOS[ScenarioType.STORM_RAIN]
        elif wind_speed > 8 and precipitation < 5:
            return PREDEFINED_SCENARIOS[ScenarioType.HIGH_WIND_SUNNY]
        elif wind_speed < 3 and humidity > 70:
            return PREDEFINED_SCENARIOS[ScenarioType.CALM_CLOUDY]
        
        # 季节性场景识别
        elif 15 <= temp <= 22 and precipitation < 10:
            return PREDEFINED_SCENARIOS[ScenarioType.NORMAL_SPRING_MILD]
        elif 25 <= temp <= 30 and humidity < 75:
            return PREDEFINED_SCENARIOS[ScenarioType.NORMAL_SUMMER_COMFORTABLE]
        elif 18 <= temp <= 25 and humidity < 60:
            return PREDEFINED_SCENARIOS[ScenarioType.NORMAL_AUTUMN_STABLE]
        elif 8 <= temp <= 15:
            return PREDEFINED_SCENARIOS[ScenarioType.NORMAL_WINTER_MILD]
        
        # 典型场景识别 (基于历史数据模式)
        elif precipitation > 10 and temp < 20:
            return PREDEFINED_SCENARIOS[ScenarioType.TYPICAL_RAINY_LOW_LOAD]
        elif temp > 25 and humidity > 75:
            return PREDEFINED_SCENARIOS[ScenarioType.TYPICAL_MILD_HUMID_HIGH_LOAD]
        
        # 默认场景
        else:
            return PREDEFINED_SCENARIOS[ScenarioType.TYPICAL_GENERAL_NORMAL]

    def _similarity_based_matching(
        self,
        weather_features: WeatherFeatures,
        load_features: Optional[Dict[str, float]] = None,
        use_enhanced_matching: bool = True
    ) -> List[Tuple[WeatherScenario, float, float]]:
        """
        基于相似度的智能匹配。
        
        Returns:
            按相似度排序的场景列表 [(scenario, similarity_score, distance), ...]
        """
        similarities = []
        
        for scenario in PREDEFINED_SCENARIOS.values():
            # 计算欧式距离
            distance = self._calculate_euclidean_distance(
                weather_features, scenario.typical_features, load_features
            )
            
            # 转换为相似度分数 (0-1，1表示完全相似)
            similarity_score = self._distance_to_similarity(distance)
            
            # 增强匹配 (考虑场景特定的调整)
            if use_enhanced_matching:
                similarity_score = self._apply_enhanced_matching(
                    similarity_score, weather_features, scenario
                )
            
            similarities.append((scenario, similarity_score, distance))
        
        # 按相似度排序 (降序)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities

    def _calculate_euclidean_distance(
        self,
        weather_features: WeatherFeatures,
        typical_features: Dict[str, float],
        load_features: Optional[Dict[str, float]] = None
    ) -> float:
        """
        计算加权欧式距离。
        """
        weather_dict = weather_features.to_dict()
        
        # 归一化处理 (避免不同量纲的影响)
        normalized_current = self._normalize_features(weather_dict)
        normalized_typical = self._normalize_features(typical_features)
        
        # 计算加权欧式距离
        total_distance = 0.0
        total_weight = 0.0
        
        for feature, weight in self.feature_weights.items():
            if feature in normalized_current and feature in normalized_typical:
                diff = normalized_current[feature] - normalized_typical[feature]
                total_distance += weight * (diff ** 2)
                total_weight += weight
        
        # 如果提供负荷特征，也纳入计算
        if load_features:
            for feature, weight in self.load_feature_weights.items():
                if feature in load_features and feature in typical_features:
                    # 假设负荷特征已经归一化
                    diff = load_features[feature] - typical_features.get(feature, 0)
                    total_distance += weight * (diff ** 2)
                    total_weight += weight
        
        # 归一化距离
        if total_weight > 0:
            return math.sqrt(total_distance / total_weight)
        else:
            return 1.0  # 最大距离

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        特征归一化 (基于气象学常见范围)。
        """
        normalization_ranges = {
            'temperature': (-20, 45),    # 温度范围 -20°C 到 45°C
            'humidity': (0, 100),        # 湿度范围 0% 到 100%
            'wind_speed': (0, 20),       # 风速范围 0 到 20 m/s
            'precipitation': (0, 100)    # 降水范围 0 到 100 mm
        }
        
        normalized = {}
        for feature, value in features.items():
            if feature in normalization_ranges:
                min_val, max_val = normalization_ranges[feature]
                # Min-Max归一化到 [0, 1]
                normalized[feature] = (value - min_val) / (max_val - min_val)
                # 限制在 [0, 1] 范围内
                normalized[feature] = max(0, min(1, normalized[feature]))
            else:
                normalized[feature] = value
        
        return normalized

    def _distance_to_similarity(self, distance: float) -> float:
        """
        将欧式距离转换为相似度分数。
        使用高斯核函数进行转换。
        """
        # 使用高斯核函数，sigma=0.5
        sigma = 0.5
        similarity = math.exp(-(distance ** 2) / (2 * sigma ** 2))
        return similarity

    def _apply_enhanced_matching(
        self,
        base_similarity: float,
        weather_features: WeatherFeatures,
        scenario: WeatherScenario
    ) -> float:
        """
        应用增强匹配算法，考虑场景特定的调整。
        """
        enhanced_similarity = base_similarity
        
        # 极端天气场景的特殊处理
        if scenario.is_extreme_weather():
            # 对于极端天气，更严格的匹配条件
            extreme_bonus = self._calculate_extreme_weather_bonus(
                weather_features, scenario
            )
            enhanced_similarity = base_similarity * (1 + extreme_bonus)
        
        # 季节性场景的时间权重 (如果有时间信息)
        seasonal_bonus = self._calculate_seasonal_bonus(scenario)
        enhanced_similarity = enhanced_similarity * (1 + seasonal_bonus)
        
        # 限制在 [0, 1] 范围内
        return max(0, min(1, enhanced_similarity))

    def _calculate_extreme_weather_bonus(
        self,
        weather_features: WeatherFeatures,
        scenario: WeatherScenario
    ) -> float:
        """
        计算极端天气场景的匹配奖励。
        """
        bonus = 0.0
        
        # 基于场景类型的特定逻辑
        if scenario.scenario_type == ScenarioType.EXTREME_HOT_HUMID:
            if weather_features.temperature > 32 and weather_features.humidity > 80:
                bonus += 0.2  # 20%奖励
        elif scenario.scenario_type == ScenarioType.EXTREME_STORM_RAIN:
            if weather_features.precipitation > 30 and weather_features.humidity > 90:
                bonus += 0.2
        elif scenario.scenario_type == ScenarioType.EXTREME_STRONG_WIND:
            if weather_features.wind_speed > 10:
                bonus += 0.2
        elif scenario.scenario_type == ScenarioType.EXTREME_HEAVY_RAIN:
            if weather_features.precipitation > 50:
                bonus += 0.2
        
        return min(bonus, 0.3)  # 最大30%奖励

    def _calculate_seasonal_bonus(self, scenario: WeatherScenario) -> float:
        """
        计算季节性奖励 (简化版，实际应用中需要当前日期信息)。
        """
        # 这里可以根据当前月份给季节性场景额外权重
        # 简化处理，给所有场景相同的基础权重
        return 0.0

    def _calculate_confidence_level(self, similarity_score: float) -> str:
        """
        计算置信度等级。
        """
        if similarity_score >= self.confidence_thresholds['high']:
            return "高"
        elif similarity_score >= self.confidence_thresholds['medium']:
            return "中"
        else:
            return "低"

    def _calculate_feature_contributions(
        self,
        weather_features: WeatherFeatures,
        typical_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算各特征对匹配结果的贡献度。
        """
        contributions = {}
        weather_dict = weather_features.to_dict()
        
        for feature in self.feature_weights:
            if feature in weather_dict and feature in typical_features:
                # 计算特征差异
                current_val = weather_dict[feature]
                typical_val = typical_features[feature]
                
                # 归一化差异
                normalized_current = self._normalize_features({feature: current_val})
                normalized_typical = self._normalize_features({feature: typical_val})
                
                diff = abs(normalized_current[feature] - normalized_typical[feature])
                
                # 贡献度 = 权重 * (1 - 差异)，差异越小贡献度越高
                contribution = self.feature_weights[feature] * (1 - diff)
                contributions[feature] = max(0, contribution)
        
        return contributions 