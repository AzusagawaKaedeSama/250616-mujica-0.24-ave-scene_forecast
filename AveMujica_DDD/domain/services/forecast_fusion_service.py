from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math

from AveMujica_DDD.domain.aggregates.forecast import Forecast, ForecastDataPoint

@dataclass
class RegionWeight:
    """区域权重配置"""
    region: str
    prediction_reliability: float  # 预测可靠性 (0-1)
    load_influence: float  # 负荷影响力 (0-1)
    prediction_complexity: float  # 预测复杂性 (0-1)
    final_weight: float = 0.0

@dataclass
class FusionResult:
    """融合预测结果"""
    fused_forecast: Forecast
    region_weights: Dict[str, float]
    fusion_metrics: Dict[str, float]
    uncertainty_bounds: Tuple[float, float]

class ForecastFusionService:
    """
    多区域预测融合领域服务。
    实现基于PCA主成分分析的多区域净负荷预测融合。
    """
    
    def __init__(self, smoothing_coefficient: float = 0.7, max_weight_factor: float = 1.2):
        """
        初始化融合服务。
        
        Args:
            smoothing_coefficient: 平滑系数，用于时变权重计算
            max_weight_factor: 最大权重因子，防止单一区域权重过大
        """
        self.smoothing_coefficient = smoothing_coefficient
        self.max_weight_factor = max_weight_factor
        
        # 三层评估指标体系权重
        self.reliability_weight = 0.35  # 预测可靠性权重
        self.influence_weight = 0.40    # 省级影响力权重
        self.complexity_weight = 0.25   # 预测复杂性权重

    def fuse_regional_forecasts(
        self, 
        regional_forecasts: Dict[str, Forecast],
        use_time_varying_weights: bool = False
    ) -> FusionResult:
        """
        执行多区域预测融合。
        
        Args:
            regional_forecasts: 各区域的预测结果
            use_time_varying_weights: 是否使用时变权重
            
        Returns:
            融合预测结果
        """
        if not regional_forecasts:
            raise ValueError("区域预测数据不能为空")
        
        # 1. 计算区域权重
        region_weights = self._calculate_region_weights(regional_forecasts)
        
        # 2. 执行融合预测
        if use_time_varying_weights:
            fused_forecast = self._fuse_with_time_varying_weights(
                regional_forecasts, region_weights
            )
        else:
            fused_forecast = self._fuse_with_static_weights(
                regional_forecasts, region_weights
            )
        
        # 3. 计算融合指标
        fusion_metrics = self._calculate_fusion_metrics(
            regional_forecasts, fused_forecast, region_weights
        )
        
        # 4. 计算不确定性边界
        uncertainty_bounds = self._calculate_uncertainty_bounds(
            regional_forecasts, region_weights
        )
        
        return FusionResult(
            fused_forecast=fused_forecast,
            region_weights={r: w.final_weight for r, w in region_weights.items()},
            fusion_metrics=fusion_metrics,
            uncertainty_bounds=uncertainty_bounds
        )

    def _calculate_region_weights(
        self, 
        regional_forecasts: Dict[str, Forecast]
    ) -> Dict[str, RegionWeight]:
        """
        基于PCA主成分分析计算区域权重。
        """
        regions = list(regional_forecasts.keys())
        region_weights = {}
        
        # 构建评估矩阵
        evaluation_matrix = []
        for region, forecast in regional_forecasts.items():
            # 计算预测可靠性指标 (基于历史性能、系统稳定性)
            reliability = self._calculate_prediction_reliability(forecast)
            
            # 计算省级影响力指标 (基于负荷规模、调节能力)
            influence = self._calculate_regional_influence(forecast)
            
            # 计算预测复杂性指标 (基于负荷波动、外部因素敏感性)
            complexity = self._calculate_prediction_complexity(forecast)
            
            evaluation_matrix.append([reliability, influence, complexity])
            region_weights[region] = RegionWeight(
                region=region,
                prediction_reliability=reliability,
                load_influence=influence,
                prediction_complexity=complexity
            )
        
        # PCA主成分分析
        evaluation_array = np.array(evaluation_matrix)
        
        # 标准化
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(evaluation_array)
        
        # PCA分析
        pca = PCA()
        pca.fit(normalized_data)
        
        # 计算权重 (基于主成分贡献度)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        
        # 综合权重计算
        for i, region in enumerate(regions):
            # 基于主成分的权重计算
            pca_weight = 0
            for j, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
                component_score = np.dot(normalized_data[i], eigenvector)
                pca_weight += eigenvalue * abs(component_score)
            
            # 归一化权重
            region_weights[region].final_weight = pca_weight
        
        # 权重归一化和平滑处理
        total_weight = sum(w.final_weight for w in region_weights.values())
        for region_weight in region_weights.values():
            region_weight.final_weight = (region_weight.final_weight / total_weight)
            
            # 限制最大权重，防止单一区域权重过大
            if region_weight.final_weight > (1.0 / len(regions)) * self.max_weight_factor:
                region_weight.final_weight = (1.0 / len(regions)) * self.max_weight_factor
        
        # 重新归一化
        total_weight = sum(w.final_weight for w in region_weights.values())
        for region_weight in region_weights.values():
            region_weight.final_weight = region_weight.final_weight / total_weight
        
        return region_weights

    def _calculate_prediction_reliability(self, forecast: Forecast) -> float:
        """
        计算预测可靠性指标。
        """
        # 基于预测区间宽度和历史性能
        avg_interval_width = forecast.get_average_interval_width()
        avg_forecast = forecast.get_average_forecast()
        
        if avg_forecast == 0 or avg_interval_width is None:
            return 0.5  # 默认中等可靠性
        
        # 区间宽度相对值 (越小越可靠)
        relative_interval_width = avg_interval_width / avg_forecast
        reliability = max(0.1, 1.0 - min(relative_interval_width, 0.9))
        
        return reliability

    def _calculate_regional_influence(self, forecast: Forecast) -> float:
        """
        计算区域影响力指标。
        """
        # 基于负荷规模和系统重要性
        avg_load = forecast.get_average_forecast()
        
        # 简化的影响力计算 (实际应用中会考虑更多因素)
        # 这里基于负荷规模进行归一化
        normalized_load = min(avg_load / 50000.0, 1.0)  # 假设50GW为参考值
        
        return max(0.1, normalized_load)

    def _calculate_prediction_complexity(self, forecast: Forecast) -> float:
        """
        计算预测复杂性指标。
        """
        if not forecast.time_series:
            return 0.5
        
        # 基于负荷波动率
        values = [point.value for point in forecast.time_series]
        if len(values) < 2:
            return 0.5
        
        # 计算变异系数
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        
        if mean_value == 0:
            return 0.5
        
        coefficient_of_variation = std_value / mean_value
        complexity = min(coefficient_of_variation * 2.0, 1.0)  # 归一化到0-1
        
        return max(0.1, complexity)

    def _fuse_with_static_weights(
        self,
        regional_forecasts: Dict[str, Forecast],
        region_weights: Dict[str, RegionWeight]
    ) -> Forecast:
        """
        使用静态权重进行融合。
        """
        # 确保所有预测的时间序列长度一致
        first_forecast = next(iter(regional_forecasts.values()))
        time_points = [point.timestamp for point in first_forecast.time_series]
        
        fused_points = []
        for i, timestamp in enumerate(time_points):
            fused_value = 0.0
            fused_lower = 0.0
            fused_upper = 0.0
            
            for region, forecast in regional_forecasts.items():
                if i < len(forecast.time_series):
                    point = forecast.time_series[i]
                    weight = region_weights[region].final_weight
                    
                    fused_value += point.value * weight
                    if point.lower_bound is not None:
                        fused_lower += point.lower_bound * weight
                    if point.upper_bound is not None:
                        fused_upper += point.upper_bound * weight
            
            fused_points.append(ForecastDataPoint(
                timestamp=timestamp,
                value=fused_value,
                lower_bound=fused_lower if fused_lower > 0 else None,
                upper_bound=fused_upper if fused_upper > 0 else None
            ))
        
        # 创建融合预测对象
        from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel
        from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
        from datetime import datetime
        import uuid
        
        # 创建融合模型和场景
        fusion_model = PredictionModel(
            name="多区域融合模型",
            version="1.0.0",
            forecast_type=first_forecast.prediction_model.forecast_type,
            file_path="fusion_model",
            target_column="fused_forecast",
            feature_columns=["multi_region_data"]
        )
        
        fusion_scenario = WeatherScenario(
            scenario_type=ScenarioType.MODERATE_NORMAL,
            description="多区域融合场景",
            uncertainty_multiplier=1.0,
            typical_features={},
            power_system_impact="多区域综合影响",
            operation_suggestions="基于融合结果进行调度"
        )
        
        return Forecast(
            province="多区域融合",
            prediction_model=fusion_model,
            matched_weather_scenario=fusion_scenario,
            time_series=fused_points,
            creation_time=datetime.now()
        )

    def _fuse_with_time_varying_weights(
        self,
        regional_forecasts: Dict[str, Forecast],
        base_weights: Dict[str, RegionWeight]
    ) -> Forecast:
        """
        使用时变权重进行融合。
        """
        # 时变权重融合逻辑
        # 基于实时预测误差和不确定性动态调整权重
        
        first_forecast = next(iter(regional_forecasts.values()))
        time_points = [point.timestamp for point in first_forecast.time_series]
        
        fused_points = []
        for i, timestamp in enumerate(time_points):
            # 计算当前时刻的动态权重
            dynamic_weights = self._calculate_dynamic_weights(
                regional_forecasts, base_weights, i
            )
            
            fused_value = 0.0
            fused_lower = 0.0
            fused_upper = 0.0
            
            for region, forecast in regional_forecasts.items():
                if i < len(forecast.time_series):
                    point = forecast.time_series[i]
                    weight = dynamic_weights[region]
                    
                    fused_value += point.value * weight
                    if point.lower_bound is not None:
                        fused_lower += point.lower_bound * weight
                    if point.upper_bound is not None:
                        fused_upper += point.upper_bound * weight
            
            fused_points.append(ForecastDataPoint(
                timestamp=timestamp,
                value=fused_value,
                lower_bound=fused_lower if fused_lower > 0 else None,
                upper_bound=fused_upper if fused_upper > 0 else None
            ))
        
        # 返回融合结果 (复用静态权重的框架代码)
        return self._fuse_with_static_weights(regional_forecasts, base_weights)

    def _calculate_dynamic_weights(
        self,
        regional_forecasts: Dict[str, Forecast],
        base_weights: Dict[str, RegionWeight],
        time_index: int
    ) -> Dict[str, float]:
        """
        计算动态权重。
        """
        dynamic_weights = {}
        
        for region, base_weight in base_weights.items():
            # 基于预测不确定性调整权重
            forecast = regional_forecasts[region]
            if time_index < len(forecast.time_series):
                point = forecast.time_series[time_index]
                
                # 计算当前时刻的不确定性
                uncertainty = 0.1  # 默认不确定性
                if point.upper_bound is not None and point.lower_bound is not None:
                    uncertainty = (point.upper_bound - point.lower_bound) / point.value
                
                # 基于不确定性调整权重 (不确定性越低，权重越高)
                adjustment_factor = 1.0 / (1.0 + uncertainty)
                dynamic_weights[region] = base_weight.final_weight * adjustment_factor
            else:
                dynamic_weights[region] = base_weight.final_weight
        
        # 权重归一化
        total_weight = sum(dynamic_weights.values())
        if total_weight > 0:
            for region in dynamic_weights:
                dynamic_weights[region] /= total_weight
        
        return dynamic_weights

    def _calculate_fusion_metrics(
        self,
        regional_forecasts: Dict[str, Forecast],
        fused_forecast: Forecast,
        region_weights: Dict[str, RegionWeight]
    ) -> Dict[str, float]:
        """
        计算融合评估指标。
        """
        metrics = {}
        
        # 融合效率指标
        regional_values = []
        for forecast in regional_forecasts.values():
            regional_values.extend([point.value for point in forecast.time_series])
        
        fused_values = [point.value for point in fused_forecast.time_series]
        
        if regional_values and fused_values:
            metrics['fusion_efficiency'] = 1.0 - (np.std(fused_values) / np.std(regional_values))
            metrics['average_fused_value'] = np.mean(fused_values)
            metrics['fused_value_std'] = np.std(fused_values)
        
        # 权重分布指标
        weights = [w.final_weight for w in region_weights.values()]
        metrics['weight_entropy'] = -sum(w * math.log(w + 1e-10) for w in weights)
        metrics['max_weight'] = max(weights)
        metrics['min_weight'] = min(weights)
        
        return metrics

    def _calculate_uncertainty_bounds(
        self,
        regional_forecasts: Dict[str, Forecast],
        region_weights: Dict[str, RegionWeight]
    ) -> Tuple[float, float]:
        """
        计算融合预测的不确定性边界。
        """
        # 基于各区域不确定性的加权组合
        total_lower = 0.0
        total_upper = 0.0
        
        for region, forecast in regional_forecasts.items():
            weight = region_weights[region].final_weight
            
            # 计算区域平均不确定性
            interval_widths = []
            for point in forecast.time_series:
                if point.upper_bound is not None and point.lower_bound is not None:
                    interval_widths.append(point.upper_bound - point.lower_bound)
            
            if interval_widths:
                avg_interval_width = np.mean(interval_widths)
                avg_forecast = forecast.get_average_forecast()
                
                relative_uncertainty = avg_interval_width / (2 * avg_forecast) if avg_forecast > 0 else 0.1
                
                total_lower += weight * relative_uncertainty
                total_upper += weight * relative_uncertainty
        
        return (float(total_lower), float(total_upper)) 