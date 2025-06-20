from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
import uuid

from AveMujica_DDD.domain.aggregates.forecast import Forecast, ForecastDataPoint
from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository
from AveMujica_DDD.domain.services.uncertainty_calculation_service import UncertaintyCalculationService
from AveMujica_DDD.domain.services.weather_scenario_recognition_service import (
    WeatherScenarioRecognitionService, WeatherFeatures, ScenarioMatchResult
)
from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider
from AveMujica_DDD.application.dtos.forecast_dto import ForecastDTO, ForecastDataPointDTO

class ScenarioAnalysisService:
    """
    场景分析应用服务。
    负责执行天气场景感知的动态不确定性预测，包括场景识别、
    不确定性调整和详细解释生成。
    """
    
    def __init__(
        self,
        forecast_repo: IForecastRepository,
        weather_scenario_repo: IWeatherScenarioRepository,
        weather_provider: IWeatherDataProvider,
        uncertainty_service: UncertaintyCalculationService,
        scenario_recognition_service: WeatherScenarioRecognitionService
    ):
        """
        初始化场景分析服务。
        """
        self._forecast_repo = forecast_repo
        self._weather_scenario_repo = weather_scenario_repo
        self._weather_provider = weather_provider
        self._uncertainty_service = uncertainty_service
        self._scenario_recognition_service = scenario_recognition_service

    def create_scenario_aware_forecast(
        self,
        province: str,
        forecast_date: date,
        base_forecast_values: List[ForecastDataPoint],
        weather_data: Optional[List[Dict]] = None
    ) -> Tuple[ForecastDTO, Dict[str, Any]]:
        """
        创建场景感知的不确定性预测。
        
        Args:
            province: 省份名称
            forecast_date: 预测日期
            base_forecast_values: 基础预测值列表
            weather_data: 天气数据 (可选，如果不提供则自动获取)
            
        Returns:
            元组：(预测结果DTO, 场景分析详情)
        """
        # 1. 获取天气数据
        if weather_data is None:
            weather_data = self._get_weather_data_for_date(province, forecast_date)
        
        # 2. 场景识别和分析
        scenario_analysis = self._analyze_weather_scenarios(weather_data)
        
        # 3. 动态不确定性调整
        adjusted_forecast_points = self._apply_scenario_aware_uncertainty(
            base_forecast_values, scenario_analysis
        )
        
        # 4. 创建预测聚合
        forecast = self._create_forecast_aggregate(
            province, scenario_analysis['dominant_scenario'], adjusted_forecast_points
        )
        
        # 5. 保存预测结果
        self._forecast_repo.save(forecast)
        
        # 6. 生成详细分析报告
        detailed_analysis = self._generate_detailed_analysis(
            scenario_analysis, weather_data, forecast
        )
        
        return self._to_dto(forecast), detailed_analysis

    def analyze_historical_scenarios(
        self,
        province: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, any]:
        """
        分析历史天气场景分布。
        
        Args:
            province: 省份名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            历史场景分析结果
        """
        # 1. 获取历史天气数据
        historical_weather = self._weather_provider.get_weather_data_for_range(
            province, start_date, end_date
        )
        
        # 2. 批量场景识别
        weather_features_list = self._convert_to_weather_features(
            historical_weather.to_dict('records')
        )
        
        scenario_results = self._scenario_recognition_service.batch_recognize_scenarios(
            weather_features_list
        )
        
        # 3. 统计分析
        statistics = self._scenario_recognition_service.get_scenario_statistics(scenario_results)
        
        # 4. 生成季节性分析
        seasonal_analysis = self._analyze_seasonal_patterns(scenario_results)
        
        # 5. 极端天气事件统计
        extreme_events = self._analyze_extreme_events(scenario_results)
        
        return {
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_days': (end_date - start_date).days + 1
            },
            'scenario_statistics': statistics,
            'seasonal_patterns': seasonal_analysis,
            'extreme_events': extreme_events,
            'scenario_details': [
                {
                    'date': start_date + timedelta(days=i),
                    'scenario': result.matched_scenario.scenario_type.value,
                    'confidence': result.confidence_level,
                    'similarity': result.similarity_score
                }
                for i, result in enumerate(scenario_results)
            ]
        }

    def evaluate_scenario_prediction_accuracy(
        self,
        province: str,
        test_dates: List[date],
        actual_load_data: Optional[Dict[date, List[float]]] = None
    ) -> Dict[str, any]:
        """
        评估场景识别和预测准确率。
        
        Args:
            province: 省份名称
            test_dates: 测试日期列表
            actual_load_data: 实际负荷数据 (可选)
            
        Returns:
            评估结果
        """
        evaluation_results = []
        
        for test_date in test_dates:
            try:
                # 获取该日期的天气数据
                weather_data = self._get_weather_data_for_date(province, test_date)
                
                # 场景识别
                scenario_analysis = self._analyze_weather_scenarios(weather_data)
                
                # 如果有实际负荷数据，计算预测准确率
                accuracy_metrics = {}
                if actual_load_data and test_date in actual_load_data:
                    actual_values = actual_load_data[test_date]
                    # 生成基础预测 (这里简化处理)
                    base_forecast = self._generate_baseline_forecast(len(actual_values))
                    
                    # 应用场景感知调整
                    adjusted_forecast = self._apply_scenario_aware_uncertainty(
                        base_forecast, scenario_analysis
                    )
                    
                    # 计算准确率指标
                    accuracy_metrics = self._calculate_accuracy_metrics(
                        actual_values, [p.value for p in adjusted_forecast]
                    )
                
                evaluation_results.append({
                    'date': test_date.isoformat(),
                    'identified_scenario': scenario_analysis['dominant_scenario']['type'],
                    'confidence': scenario_analysis['dominant_scenario']['confidence'],
                    'similarity_score': scenario_analysis['dominant_scenario']['similarity'],
                    'accuracy_metrics': accuracy_metrics,
                    'success': True
                })
                
            except Exception as e:
                evaluation_results.append({
                    'date': test_date.isoformat(),
                    'error': str(e),
                    'success': False
                })
        
        # 汇总评估结果
        successful_results = [r for r in evaluation_results if r['success']]
        
        return {
            'evaluation_summary': {
                'total_tests': len(test_dates),
                'successful_tests': len(successful_results),
                'success_rate': len(successful_results) / len(test_dates) if test_dates else 0,
                'average_scenario_confidence': sum(r['confidence'] for r in successful_results) / len(successful_results) if successful_results else 0,
                'average_similarity_score': sum(r['similarity_score'] for r in successful_results) / len(successful_results) if successful_results else 0
            },
            'detailed_results': evaluation_results,
            'scenario_accuracy_by_type': self._calculate_scenario_accuracy_by_type(successful_results)
        }

    def _analyze_weather_scenarios(self, weather_data: List[Dict]) -> Dict[str, any]:
        """
        分析天气场景。
        """
        if not weather_data:
            raise ValueError("天气数据不能为空")
        
        # 转换为WeatherFeatures格式
        weather_features_list = self._convert_to_weather_features(weather_data)
        
        # 批量场景识别
        scenario_results = self._scenario_recognition_service.batch_recognize_scenarios(
            weather_features_list
        )
        
        # 统计分析
        statistics = self._scenario_recognition_service.get_scenario_statistics(scenario_results)
        
        # 确定主导场景
        dominant_scenario_info = statistics.get('dominant_scenario', {})
        dominant_scenario = None
        
        if scenario_results:
            # 使用第一个结果作为主导场景的详细信息
            first_result = scenario_results[0]
            dominant_scenario = {
                'type': first_result.matched_scenario.scenario_type.value,
                'description': first_result.matched_scenario.description,
                'confidence': first_result.confidence_level,
                'similarity': first_result.similarity_score,
                'uncertainty_multiplier': first_result.matched_scenario.uncertainty_multiplier,
                'operation_suggestions': first_result.matched_scenario.operation_suggestions,
                'power_system_impact': first_result.matched_scenario.power_system_impact,
                'scenario_object': first_result.matched_scenario
            }
        
        return {
            'total_samples': len(weather_features_list),
            'scenario_statistics': statistics,
            'dominant_scenario': dominant_scenario,
            'all_scenario_results': scenario_results
        }

    def _apply_scenario_aware_uncertainty(
        self,
        base_forecast_points: List[ForecastDataPoint],
        scenario_analysis: Dict[str, any]
    ) -> List[ForecastDataPoint]:
        """
        应用场景感知的动态不确定性调整。
        """
        dominant_scenario = scenario_analysis['dominant_scenario']['scenario_object']
        
        adjusted_points = []
        for point in base_forecast_points:
            # 使用不确定性计算服务进行动态调整
            lower_bound, upper_bound = self._uncertainty_service.calculate_bounds(
                predicted_value=point.value,
                base_uncertainty_rate=0.05,  # 基础不确定性5%
                scenario=dominant_scenario
            )
            
            adjusted_points.append(ForecastDataPoint(
                timestamp=point.timestamp,
                value=point.value,
                lower_bound=lower_bound,
                upper_bound=upper_bound
            ))
        
        return adjusted_points

    def _create_forecast_aggregate(
        self,
        province: str,
        dominant_scenario: Dict[str, any],
        forecast_points: List[ForecastDataPoint]
    ) -> Forecast:
        """
        创建预测聚合对象。
        """
        from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
        
        # 创建场景感知模型
        scenario_aware_model = PredictionModel(
            name=f"{province}_场景感知模型",
            version="1.0.0",
            forecast_type=ForecastType.LOAD,
            file_path="scenario_aware_model",
            target_column="load",
            feature_columns=["weather_scenario", "temperature", "humidity", "wind_speed", "precipitation"]
        )
        
        return Forecast(
            province=province,
            prediction_model=scenario_aware_model,
            matched_weather_scenario=dominant_scenario['scenario_object'],
            time_series=forecast_points,
            creation_time=datetime.now()
        )

    def _generate_detailed_analysis(
        self,
        scenario_analysis: Dict[str, any],
        weather_data: List[Dict],
        forecast: Forecast
    ) -> Dict[str, any]:
        """
        生成详细的场景分析报告。
        """
        dominant_scenario = scenario_analysis['dominant_scenario']
        
        # 不确定性来源分析
        uncertainty_sources = self._analyze_uncertainty_sources(dominant_scenario)
        
        # 风险评估
        risk_assessment = self._assess_power_system_risks(dominant_scenario)
        
        # 操作建议
        operation_recommendations = self._generate_operation_recommendations(
            dominant_scenario, forecast
        )
        
        return {
            'scenario_identification': {
                'identified_scenario': dominant_scenario['type'],
                'confidence_level': dominant_scenario['confidence'],
                'similarity_score': dominant_scenario['similarity'],
                'description': dominant_scenario['description']
            },
            'uncertainty_analysis': {
                'base_uncertainty_rate': 0.05,
                'scenario_multiplier': dominant_scenario['uncertainty_multiplier'],
                'final_uncertainty_rate': 0.05 * dominant_scenario['uncertainty_multiplier'],
                'uncertainty_sources': uncertainty_sources,
                'average_interval_width': forecast.get_average_interval_width()
            },
            'power_system_impact': {
                'impact_description': dominant_scenario['power_system_impact'],
                'risk_level': risk_assessment['risk_level'],
                'key_concerns': risk_assessment['key_concerns']
            },
            'operation_recommendations': operation_recommendations,
            'weather_conditions': {
                'sample_count': len(weather_data),
                'temperature_range': self._get_weather_range(weather_data, 'temperature'),
                'humidity_range': self._get_weather_range(weather_data, 'humidity'),
                'wind_speed_range': self._get_weather_range(weather_data, 'wind_speed'),
                'precipitation_total': sum(d.get('precipitation', 0) for d in weather_data)
            },
            'forecast_summary': {
                'prediction_points': len(forecast.time_series),
                'average_forecast': forecast.get_average_forecast(),
                'forecast_range': {
                    'min': min(p.value for p in forecast.time_series),
                    'max': max(p.value for p in forecast.time_series)
                }
            }
        }

    def _convert_to_weather_features(self, weather_data: List[Dict]) -> List[WeatherFeatures]:
        """
        将天气数据转换为WeatherFeatures格式。
        """
        features_list = []
        for data in weather_data:
            features = WeatherFeatures(
                temperature=data.get('temperature', 20.0),
                humidity=data.get('humidity', 60.0),
                wind_speed=data.get('wind_speed', 4.0),
                precipitation=data.get('precipitation', 0.0)
            )
            features_list.append(features)
        
        return features_list

    def _get_weather_data_for_date(self, province: str, forecast_date: date) -> List[Dict]:
        """
        获取指定日期的天气数据。
        """
        # 这里简化处理，实际应用中会调用天气数据提供者
        # 返回模拟的天气数据
        return [{
            'temperature': 25.0,
            'humidity': 70.0,
            'wind_speed': 4.0,
            'precipitation': 0.0,
            'time': datetime.combine(forecast_date, datetime.min.time())
        }]

    def _generate_baseline_forecast(self, num_points: int) -> List[ForecastDataPoint]:
        """
        生成基线预测 (简化实现)。
        """
        from datetime import timedelta
        base_time = datetime.now()
        
        return [
            ForecastDataPoint(
                timestamp=base_time + timedelta(hours=i),
                value=25000 + i * 100,  # 简化的负荷模式
                lower_bound=None,
                upper_bound=None
            )
            for i in range(num_points)
        ]

    def _calculate_accuracy_metrics(
        self, 
        actual_values: List[float], 
        predicted_values: List[float]
    ) -> Dict[str, float]:
        """
        计算预测准确率指标。
        """
        if len(actual_values) != len(predicted_values):
            return {}
        
        import numpy as np
        
        actual = np.array(actual_values)
        predicted = np.array(predicted_values)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actual - predicted))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # 相关系数
        correlation = np.corrcoef(actual, predicted)[0, 1]
        
        return {
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),
            'correlation': float(correlation)
        }

    def _analyze_uncertainty_sources(self, dominant_scenario: Dict[str, any]) -> List[str]:
        """
        分析不确定性来源。
        """
        sources = []
        
        # 基于场景类型分析不确定性来源
        scenario_type = dominant_scenario['type']
        
        if '极端' in scenario_type:
            sources.append("极端天气条件导致的负荷模式异常")
            sources.append("历史数据中极端天气样本有限")
        
        if '高温' in scenario_type:
            sources.append("空调负荷的非线性响应")
            sources.append("用户行为的随机性增加")
        
        if '风' in scenario_type:
            sources.append("风电出力的随机波动")
            sources.append("系统调节能力的限制")
        
        if '雨' in scenario_type or '降水' in scenario_type:
            sources.append("天气预报误差的影响")
            sources.append("设备故障概率增加")
        
        # 默认不确定性来源
        if not sources:
            sources.append("负荷预测模型的固有误差")
            sources.append("未来天气条件的不确定性")
        
        return sources

    def _assess_power_system_risks(self, dominant_scenario: Dict[str, any]) -> Dict[str, any]:
        """
        评估电力系统风险。
        """
        uncertainty_multiplier = dominant_scenario['uncertainty_multiplier']
        
        if uncertainty_multiplier >= 3.0:
            risk_level = "高"
            key_concerns = ["系统稳定性", "供电安全", "设备过载", "应急响应"]
        elif uncertainty_multiplier >= 2.0:
            risk_level = "中高"
            key_concerns = ["负荷调度", "备用容量", "设备监控"]
        elif uncertainty_multiplier >= 1.5:
            risk_level = "中等"
            key_concerns = ["经济调度", "负荷跟踪"]
        else:
            risk_level = "低"
            key_concerns = ["正常运行监控"]
        
        return {
            'risk_level': risk_level,
            'key_concerns': key_concerns
        }

    def _generate_operation_recommendations(
        self,
        dominant_scenario: Dict[str, any],
        forecast: Forecast
    ) -> List[str]:
        """
        生成运行建议。
        """
        recommendations = []
        
        # 基于场景的基础建议
        base_suggestions = dominant_scenario.get('operation_suggestions', '')
        if base_suggestions:
            recommendations.append(base_suggestions)
        
        # 基于备用容量建议
        scenario_obj = dominant_scenario['scenario_object']
        backup_capacity = scenario_obj.get_backup_capacity_recommendation()
        recommendations.append(f"建议配置{backup_capacity*100:.0f}%的备用容量")
        
        # 基于预测区间宽度的建议
        avg_interval_width = forecast.get_average_interval_width()
        if avg_interval_width and avg_interval_width > 5000:  # 假设5GW为高不确定性阈值
            recommendations.append("预测不确定性较高，建议增强实时监控和快速响应能力")
        
        return recommendations

    def _get_weather_range(self, weather_data: List[Dict], field: str) -> Dict[str, float]:
        """
        获取天气要素的范围。
        """
        values = [d.get(field, 0) for d in weather_data if field in d]
        if values:
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values)
            }
        return {'min': 0, 'max': 0, 'avg': 0}

    def _analyze_seasonal_patterns(self, scenario_results: List[ScenarioMatchResult]) -> Dict[str, any]:
        """
        分析季节性模式 (简化实现)。
        """
        # 这里简化处理，实际应用中需要日期信息
        return {
            'seasonal_distribution': '需要日期信息进行季节性分析',
            'peak_seasons': [],
            'stable_seasons': []
        }

    def _analyze_extreme_events(self, scenario_results: List[ScenarioMatchResult]) -> Dict[str, any]:
        """
        分析极端天气事件。
        """
        extreme_events = []
        total_extreme = 0
        
        for result in scenario_results:
            if result.matched_scenario.is_extreme_weather():
                total_extreme += 1
                extreme_events.append({
                    'scenario_type': result.matched_scenario.scenario_type.value,
                    'confidence': result.confidence_level,
                    'uncertainty_multiplier': result.matched_scenario.uncertainty_multiplier
                })
        
        return {
            'total_extreme_events': total_extreme,
            'extreme_event_rate': total_extreme / len(scenario_results) if scenario_results else 0,
            'extreme_events_detail': extreme_events
        }

    def _calculate_scenario_accuracy_by_type(self, successful_results: List[Dict]) -> Dict[str, any]:
        """
        按场景类型计算识别准确率。
        """
        scenario_stats = {}
        
        for result in successful_results:
            scenario_type = result['identified_scenario']
            if scenario_type not in scenario_stats:
                scenario_stats[scenario_type] = {
                    'count': 0,
                    'total_confidence': 0,
                    'total_similarity': 0
                }
            
            stats = scenario_stats[scenario_type]
            stats['count'] += 1
            stats['total_confidence'] += result['confidence']
            stats['total_similarity'] += result['similarity_score']
        
        # 计算平均值
        for scenario_type, stats in scenario_stats.items():
            count = stats['count']
            stats['average_confidence'] = stats['total_confidence'] / count
            stats['average_similarity'] = stats['total_similarity'] / count
            stats['percentage'] = count / len(successful_results) * 100
        
        return scenario_stats

    def _to_dto(self, forecast: Forecast) -> ForecastDTO:
        """
        将Forecast聚合转换为DTO。
        """
        return ForecastDTO(
            forecast_id=forecast.forecast_id,
            province=forecast.province,
            creation_time=forecast.creation_time,
            model_name=forecast.prediction_model.name,
            scenario_type=forecast.matched_weather_scenario.scenario_type.value,
            time_series=[
                ForecastDataPointDTO(
                    timestamp=p.timestamp,
                    value=p.value,
                    upper_bound=p.upper_bound,
                    lower_bound=p.lower_bound
                ) for p in forecast.time_series
            ]
        ) 