import uuid
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from AveMujica_DDD.domain.aggregates.forecast import Forecast, ForecastDataPoint
from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository
from AveMujica_DDD.domain.services.uncertainty_calculation_service import UncertaintyCalculationService
from AveMujica_DDD.domain.services.forecast_fusion_service import ForecastFusionService
from AveMujica_DDD.domain.services.weather_scenario_recognition_service import (
    WeatherScenarioRecognitionService, WeatherFeatures
)
from AveMujica_DDD.application.ports.i_prediction_engine import IPredictionEngine
from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider
from AveMujica_DDD.application.dtos.forecast_dto import ForecastDTO, ForecastDataPointDTO


class ForecastService:
    """
    应用服务：负责编排和执行预测相关的用例。
    支持多种预测模式：日前、区间、概率、滚动预测。
    支持多源预测：负荷、光伏、风电。
    """
    def __init__(
        self,
        forecast_repo: IForecastRepository,
        model_repo: IModelRepository,
        weather_scenario_repo: IWeatherScenarioRepository,
        weather_provider: IWeatherDataProvider,
        prediction_engine: IPredictionEngine,
        uncertainty_service: UncertaintyCalculationService,
        fusion_service: ForecastFusionService,
        scenario_recognition_service: WeatherScenarioRecognitionService,
    ):
        """
        通过依赖注入，将所有依赖项（仓储、外部端口、领域服务）传入。
        """
        self._forecast_repo = forecast_repo
        self._model_repo = model_repo
        self._weather_scenario_repo = weather_scenario_repo
        self._weather_provider = weather_provider
        self._prediction_engine = prediction_engine
        self._uncertainty_service = uncertainty_service
        self._fusion_service = fusion_service
        self._scenario_recognition_service = scenario_recognition_service

    # ==== API便捷方法 ====
    
    def create_day_ahead_load_forecast(
        self,
        province: str,
        start_date: date,
        end_date: date,
        model_id: Optional[uuid.UUID] = None,
        historical_days: int = 7,
        weather_aware: bool = True
    ) -> ForecastDTO:
        """
        API便捷方法：创建日前负荷预测
        """
        return self.create_day_ahead_forecast(
            province=province,
            start_date=start_date,
            end_date=end_date,
            forecast_type=ForecastType.LOAD,
            model_id=model_id,
            historical_days=historical_days,
            weather_aware=weather_aware
        )
    
    def create_day_ahead_pv_forecast(
        self,
        province: str,
        start_date: date,
        end_date: date,
        model_id: Optional[uuid.UUID] = None,
        historical_days: int = 7,
        weather_aware: bool = True
    ) -> ForecastDTO:
        """
        API便捷方法：创建日前光伏预测
        """
        return self.create_day_ahead_forecast(
            province=province,
            start_date=start_date,
            end_date=end_date,
            forecast_type=ForecastType.PV,
            model_id=model_id,
            historical_days=historical_days,
            weather_aware=weather_aware
        )
    
    def create_day_ahead_wind_forecast(
        self,
        province: str,
        start_date: date,
        end_date: date,
        model_id: Optional[uuid.UUID] = None,
        historical_days: int = 7,
        weather_aware: bool = True
    ) -> ForecastDTO:
        """
        API便捷方法：创建日前风电预测
        """
        return self.create_day_ahead_forecast(
            province=province,
            start_date=start_date,
            end_date=end_date,
            forecast_type=ForecastType.WIND,
            model_id=model_id,
            historical_days=historical_days,
            weather_aware=weather_aware
        )

    # ==== 核心预测方法 ====

    def create_day_ahead_forecast(
        self, 
        province: str, 
        start_date: date,
        end_date: date,
        forecast_type: ForecastType,
        model_id: Optional[uuid.UUID] = None,
        historical_days: int = 7,
        weather_aware: bool = False
    ) -> ForecastDTO:
        """
        创建日前预测（负荷/光伏/风电）。
        
        Args:
            province: 省份名称
            start_date: 预测开始日期
            end_date: 预测结束日期
            forecast_type: 预测类型（负荷/光伏/风电）
            model_id: 指定模型ID (可选)
            historical_days: 历史数据天数
            weather_aware: 是否启用天气感知预测
            
        Returns:
            预测结果DTO
        """
        # 1. 获取或创建模型
        model = self._get_or_create_model(province, forecast_type, model_id)

        # 2. 获取历史和预测期间的数据
        hist_start_date = start_date - timedelta(days=historical_days)
        combined_data = self._weather_provider.get_weather_data_for_range(
            province, hist_start_date, end_date
        )

        # 3. 调用预测引擎
        point_forecast_series = self._prediction_engine.predict(model, combined_data)
        
        # 4. 选择目标日期范围的预测结果
        target_forecast_series = self._extract_target_forecast(
            point_forecast_series, start_date, end_date
        )

        # 5. 场景识别和不确定性计算
        scenario, forecast_points = self._process_scenario_and_uncertainty(
            combined_data, target_forecast_series, start_date, end_date, weather_aware
        )
        
        # 6. 创建并保存预测聚合
        forecast = Forecast(
            province=province,
            prediction_model=model,
            matched_weather_scenario=scenario,
            time_series=forecast_points,
            creation_time=datetime.now()
        )
        
        self._forecast_repo.save(forecast)
        return self._to_dto(forecast)

    def create_interval_forecast(
        self,
        province: str,
        start_date: date,
        end_date: date,
        forecast_type: ForecastType,
        confidence_level: float = 0.95,
        model_id: Optional[uuid.UUID] = None,
        weather_aware: bool = True
    ) -> ForecastDTO:
        """
        创建区间预测。
        
        Args:
            province: 省份名称
            start_date: 预测开始日期
            end_date: 预测结束日期
            forecast_type: 预测类型
            confidence_level: 置信水平 (0.8, 0.9, 0.95, 0.99)
            model_id: 指定模型ID (可选)
            weather_aware: 是否启用天气感知
            
        Returns:
            带置信区间的预测结果
        """
        # 1. 执行点预测
        point_forecast = self.create_day_ahead_forecast(
            province, start_date, end_date, forecast_type, model_id, weather_aware=weather_aware
        )
        
        # 2. 计算置信区间
        interval_forecast_points = self._calculate_confidence_intervals(
            point_forecast.time_series, confidence_level, weather_aware
        )
        
        # 3. 创建区间预测聚合
        model = self._get_or_create_model(province, forecast_type, model_id)
        scenario = self._weather_scenario_repo.find_by_type(ScenarioType.MODERATE_NORMAL)
        
        forecast = Forecast(
            province=province,
            prediction_model=model,
            matched_weather_scenario=scenario,
            time_series=interval_forecast_points,
            creation_time=datetime.now()
        )
        
        self._forecast_repo.save(forecast)
        return self._to_dto(forecast)

    def create_probabilistic_forecast(
        self,
        province: str,
        start_date: date,
        end_date: date,
        forecast_type: ForecastType,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        model_id: Optional[uuid.UUID] = None
    ) -> Dict[str, ForecastDTO]:
        """
        创建概率预测。
        
        Args:
            province: 省份名称
            start_date: 预测开始日期
            end_date: 预测结束日期
            forecast_type: 预测类型
            quantiles: 分位数列表
            model_id: 指定模型ID (可选)
            
        Returns:
            分位数预测结果字典
        """
        # 1. 获取基础预测
        base_forecast = self.create_day_ahead_forecast(
            province, start_date, end_date, forecast_type, model_id
        )
        
        # 2. 为每个分位数生成预测
        quantile_forecasts = {}
        
        for quantile in quantiles:
            quantile_points = self._calculate_quantile_forecast(
                base_forecast.time_series, quantile
            )
            
            # 创建分位数预测聚合
            model = self._get_or_create_model(province, forecast_type, model_id)
            scenario = self._weather_scenario_repo.find_by_type(ScenarioType.MODERATE_NORMAL)
            
            forecast = Forecast(
                province=province,
                prediction_model=model,
                matched_weather_scenario=scenario,
                time_series=quantile_points,
                creation_time=datetime.now()
            )
            
            self._forecast_repo.save(forecast)
            quantile_forecasts[f"q{quantile}"] = self._to_dto(forecast)
        
        return quantile_forecasts

    def create_rolling_forecast(
        self,
        province: str,
        start_date: date,
        end_date: date,
        forecast_type: ForecastType,
        forecast_horizon: int = 24,  # 小时
        update_interval: int = 1,    # 小时
        model_id: Optional[uuid.UUID] = None
    ) -> List[ForecastDTO]:
        """
        创建滚动预测。
        
        Args:
            province: 省份名称
            start_date: 预测开始日期
            end_date: 预测结束日期
            forecast_type: 预测类型
            forecast_horizon: 预测时域（小时）
            update_interval: 更新间隔（小时）
            model_id: 指定模型ID (可选)
            
        Returns:
            滚动预测结果列表
        """
        rolling_forecasts = []
        current_time = datetime.combine(start_date, datetime.min.time())
        end_time = datetime.combine(end_date, datetime.max.time())
        
        while current_time < end_time:
            # 计算当前预测的结束时间
            forecast_end_time = current_time + timedelta(hours=forecast_horizon)
            forecast_end_date = forecast_end_time.date()
            
            # 执行当前时刻的预测
            try:
                current_forecast = self.create_day_ahead_forecast(
                    province, current_time.date(), forecast_end_date, 
                    forecast_type, model_id
                )
                rolling_forecasts.append(current_forecast)
            except Exception as e:
                # 记录错误但继续执行
                print(f"滚动预测错误 {current_time}: {e}")
            
            # 更新到下一个时刻
            current_time += timedelta(hours=update_interval)
        
        return rolling_forecasts

    def create_multi_regional_fusion_forecast(
        self,
        regional_provinces: List[str],
        start_date: date,
        end_date: date,
        forecast_type: ForecastType = ForecastType.LOAD,
        use_time_varying_weights: bool = False
    ) -> Tuple[ForecastDTO, Dict[str, Any]]:
        """
        创建多区域融合预测。
        
        Args:
            regional_provinces: 参与融合的省份列表
            start_date: 预测开始日期
            end_date: 预测结束日期
            forecast_type: 预测类型
            use_time_varying_weights: 是否使用时变权重
            
        Returns:
            元组：(融合预测结果, 融合详情)
        """
        # 1. 为每个省份执行预测
        regional_forecasts = {}
        for province in regional_provinces:
            try:
                forecast_dto = self.create_day_ahead_forecast(
                    province, start_date, end_date, forecast_type
                )
                # 转换为Forecast聚合对象
                forecast = self._dto_to_forecast(forecast_dto, province)
                regional_forecasts[province] = forecast
            except Exception as e:
                print(f"省份 {province} 预测失败: {e}")
        
        if not regional_forecasts:
            raise ValueError("没有成功的区域预测，无法执行融合")
        
        # 2. 执行融合
        fusion_result = self._fusion_service.fuse_regional_forecasts(
            regional_forecasts, use_time_varying_weights
        )
        
        # 3. 保存融合结果
        self._forecast_repo.save(fusion_result.fused_forecast)
        
        # 4. 构建详细信息
        fusion_details = {
            'participating_regions': list(regional_forecasts.keys()),
            'region_weights': fusion_result.region_weights,
            'fusion_metrics': fusion_result.fusion_metrics,
            'uncertainty_bounds': fusion_result.uncertainty_bounds,
            'fusion_method': 'PCA主成分分析',
            'time_varying_weights': use_time_varying_weights
        }
        
        return self._to_dto(fusion_result.fused_forecast), fusion_details

    def create_net_load_forecast(
        self,
        province: str,
        start_date: date,
        end_date: date,
        include_pv: bool = True,
        include_wind: bool = True,
        model_id: Optional[uuid.UUID] = None
    ) -> Tuple[ForecastDTO, Dict[str, ForecastDTO]]:
        """
        创建净负荷预测 (总负荷 - 新能源出力)。
        
        Args:
            province: 省份名称
            start_date: 预测开始日期
            end_date: 预测结束日期
            include_pv: 是否包含光伏预测
            include_wind: 是否包含风电预测
            model_id: 指定模型ID (可选)
            
        Returns:
            元组：(净负荷预测, 各分项预测字典)
        """
        # 1. 总负荷预测
        load_forecast = self.create_day_ahead_forecast(
            province, start_date, end_date, ForecastType.LOAD, model_id
        )
        
        component_forecasts = {'load': load_forecast}
        
        # 2. 光伏预测
        pv_forecast = None
        if include_pv:
            try:
                pv_forecast = self.create_day_ahead_forecast(
                    province, start_date, end_date, ForecastType.PV, model_id
                )
                component_forecasts['pv'] = pv_forecast
            except Exception as e:
                print(f"光伏预测失败: {e}")
        
        # 3. 风电预测
        wind_forecast = None
        if include_wind:
            try:
                wind_forecast = self.create_day_ahead_forecast(
                    province, start_date, end_date, ForecastType.WIND, model_id
                )
                component_forecasts['wind'] = wind_forecast
            except Exception as e:
                print(f"风电预测失败: {e}")
        
        # 4. 计算净负荷
        net_load_points = self._calculate_net_load(
            load_forecast.time_series, pv_forecast, wind_forecast
        )
        
        # 5. 创建净负荷预测聚合
        model = self._get_or_create_model(province, ForecastType.LOAD, model_id)
        scenario = self._weather_scenario_repo.find_by_type(ScenarioType.MODERATE_NORMAL)
        
        net_load_forecast = Forecast(
            province=province,
            prediction_model=model,
            matched_weather_scenario=scenario,
            time_series=net_load_points,
            creation_time=datetime.now()
        )
        
        self._forecast_repo.save(net_load_forecast)
        
        return self._to_dto(net_load_forecast), component_forecasts

    def _get_or_create_model(
        self, 
        province: str, 
        forecast_type: ForecastType, 
        model_id: Optional[uuid.UUID] = None
    ) -> PredictionModel:
        """
        获取或创建预测模型。
        """
        if model_id:
            model = self._model_repo.find_by_id(model_id)
            if model and model.forecast_type == forecast_type:
                return model
        
        # 查找现有模型
        existing_models = self._model_repo.find_by_type_and_region(forecast_type, province)
        if existing_models:
            return existing_models[0]
        
        # 创建新模型
        model = PredictionModel(
            name=f"{province}_{forecast_type.value}_模型",
            version="1.0.0",
            forecast_type=forecast_type,
            file_path=f"models/{forecast_type.value.lower()}/{province}",
            target_column=forecast_type.value.lower(),
            feature_columns=["temperature", "humidity", "wind_speed", "precipitation"]
        )
        
        self._model_repo.save(model)
        return model

    def _extract_target_forecast(
        self,
        forecast_series: pd.Series,
        start_date: date,
        end_date: date
    ) -> pd.Series:
        """
        从预测序列中提取目标日期范围的数据。
        """
        target_start = datetime.combine(start_date, datetime.min.time())
        target_end = datetime.combine(end_date, datetime.max.time())
        
        return forecast_series.loc[target_start:target_end]

    def _process_scenario_and_uncertainty(
        self,
        combined_data: pd.DataFrame,
        target_forecast_series: pd.Series,
        start_date: date,
        end_date: date,
        weather_aware: bool
    ) -> Tuple[WeatherScenario, List[ForecastDataPoint]]:
        """
        处理场景识别和不确定性计算。
        """
        target_start = datetime.combine(start_date, datetime.min.time())
        target_end = datetime.combine(end_date, datetime.max.time())
        target_weather_data = combined_data.loc[target_start:target_end]
        
        # 场景识别
        if weather_aware and not target_weather_data.empty:
            weather_data_list = target_weather_data.to_dict('records')
            scenario = self._recognize_scenario(weather_data_list)
        else:
            scenario = self._weather_scenario_repo.find_by_type(ScenarioType.MODERATE_NORMAL)
        
        # 创建预测数据点
        forecast_points = []
        for timestamp, value in target_forecast_series.items():
            lower, upper = self._uncertainty_service.calculate_bounds(
                predicted_value=value,
                base_uncertainty_rate=self._get_base_uncertainty_rate(scenario),
                scenario=scenario
            )
            forecast_points.append(
                ForecastDataPoint(
                    timestamp=timestamp, 
                    value=value, 
                    lower_bound=lower, 
                    upper_bound=upper
                )
            )
        
        return scenario, forecast_points

    def _recognize_scenario(self, weather_data: list) -> WeatherScenario:
        """
        场景识别逻辑。
        """
        if not weather_data:
            return self._weather_scenario_repo.find_by_type(ScenarioType.MODERATE_NORMAL)

        # 转换为WeatherFeatures
        avg_weather = {
            'temperature': sum(d.get('temperature', 20) for d in weather_data) / len(weather_data),
            'humidity': sum(d.get('humidity', 60) for d in weather_data) / len(weather_data),
            'wind_speed': sum(d.get('wind_speed', 4) for d in weather_data) / len(weather_data),
            'precipitation': sum(d.get('precipitation', 0) for d in weather_data) / len(weather_data)
        }
        
        weather_features = WeatherFeatures(**avg_weather)
        
        # 使用场景识别服务
        match_result = self._scenario_recognition_service.recognize_scenario(weather_features)
        
        return match_result.matched_scenario

    def _get_base_uncertainty_rate(self, scenario: WeatherScenario) -> float:
        """
        根据预测类型获取基础不确定性率。
        """
        # 不同预测类型的基础不确定性
        base_rates = {
            ForecastType.LOAD: 0.05,  # 负荷预测 5%
            ForecastType.PV: 0.15,    # 光伏预测 15%
            ForecastType.WIND: 0.20   # 风电预测 20%
        }
        
        return base_rates.get(ForecastType.LOAD, 0.05)

    def _calculate_confidence_intervals(
        self,
        time_series: List[ForecastDataPointDTO],
        confidence_level: float,
        weather_aware: bool
    ) -> List[ForecastDataPoint]:
        """
        计算置信区间。
        """
        # Z值对应表
        z_values = {
            0.80: 1.28,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        z = z_values.get(confidence_level, 1.96)
        
        interval_points = []
        for point_dto in time_series:
            # 估算标准差 (简化处理)
            estimated_std = point_dto.value * 0.1  # 假设变异系数为10%
            
            # 计算置信区间
            margin = z * estimated_std
            lower_bound = point_dto.value - margin
            upper_bound = point_dto.value + margin
            
            interval_points.append(ForecastDataPoint(
                timestamp=point_dto.timestamp,
                value=point_dto.value,
                lower_bound=max(0, lower_bound),  # 确保非负
                upper_bound=upper_bound
            ))
        
        return interval_points

    def _calculate_quantile_forecast(
        self,
        time_series: List[ForecastDataPointDTO],
        quantile: float
    ) -> List[ForecastDataPoint]:
        """
        计算分位数预测。
        """
        quantile_points = []
        for point_dto in time_series:
            # 简化的分位数计算
            if quantile <= 0.5:
                # 下分位数
                quantile_value = point_dto.value * (0.8 + 0.4 * quantile)
            else:
                # 上分位数
                quantile_value = point_dto.value * (0.8 + 0.4 * quantile)
            
            quantile_points.append(ForecastDataPoint(
                timestamp=point_dto.timestamp,
                value=quantile_value,
                lower_bound=None,
                upper_bound=None
            ))
        
        return quantile_points

    def _calculate_net_load(
        self,
        load_forecast: List[ForecastDataPointDTO],
        pv_forecast: Optional[ForecastDTO],
        wind_forecast: Optional[ForecastDTO]
    ) -> List[ForecastDataPoint]:
        """
        计算净负荷 = 总负荷 - 光伏出力 - 风电出力。
        """
        net_load_points = []
        
        for i, load_point in enumerate(load_forecast):
            net_load = load_point.value
            
            # 减去光伏出力
            if pv_forecast and i < len(pv_forecast.time_series):
                pv_value = pv_forecast.time_series[i].value
                net_load -= pv_value
            
            # 减去风电出力
            if wind_forecast and i < len(wind_forecast.time_series):
                wind_value = wind_forecast.time_series[i].value
                net_load -= wind_value
            
            # 确保净负荷非负
            net_load = max(0, net_load)
            
            net_load_points.append(ForecastDataPoint(
                timestamp=load_point.timestamp,
                value=net_load,
                lower_bound=load_point.lower_bound,
                upper_bound=load_point.upper_bound
            ))
        
        return net_load_points

    def _dto_to_forecast(self, forecast_dto: ForecastDTO, province: str) -> Forecast:
        """
        将DTO转换为Forecast聚合对象。
        """
        # 获取模型和场景 (简化处理)
        model = self._get_or_create_model(province, ForecastType.LOAD)
        scenario = self._weather_scenario_repo.find_by_type(ScenarioType.MODERATE_NORMAL)
        
        # 转换时间序列
        time_series = [
            ForecastDataPoint(
                timestamp=point.timestamp,
                value=point.value,
                lower_bound=point.lower_bound,
                upper_bound=point.upper_bound
            )
            for point in forecast_dto.time_series
        ]
        
        return Forecast(
            forecast_id=forecast_dto.forecast_id,
            province=province,
            prediction_model=model,
            matched_weather_scenario=scenario,
            time_series=time_series,
            creation_time=forecast_dto.creation_time
        )

    def _to_dto(self, forecast: Forecast) -> ForecastDTO:
        """将Forecast聚合转换为ForecastDTO。"""
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

    def _get_forecast_type_from_string(self, forecast_type_str: str) -> ForecastType:
        """
        从字符串获取ForecastType枚举值。
        
        Args:
            forecast_type_str: 预测类型字符串 ('load', 'pv', 'wind')
            
        Returns:
            对应的ForecastType枚举值
        """
        forecast_type_mapping = {
            'load': ForecastType.LOAD,
            'pv': ForecastType.PV,
            'wind': ForecastType.WIND
        }
        
        return forecast_type_mapping.get(forecast_type_str.lower(), ForecastType.LOAD) 