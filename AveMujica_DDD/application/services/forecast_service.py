import uuid
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

from AveMujica_DDD.domain.aggregates.forecast import Forecast, ForecastDataPoint
from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository
from AveMujica_DDD.domain.services.uncertainty_calculation_service import UncertaintyCalculationService
from AveMujica_DDD.application.ports.i_prediction_engine import IPredictionEngine
from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider
from AveMujica_DDD.application.dtos.forecast_dto import ForecastDTO, ForecastDataPointDTO


class ForecastService:
    """
    应用服务：负责编排和执行预测相关的用例。
    """
    def __init__(
        self,
        forecast_repo: IForecastRepository,
        model_repo: IModelRepository,
        weather_scenario_repo: IWeatherScenarioRepository,
        weather_provider: IWeatherDataProvider,
        prediction_engine: IPredictionEngine,
        uncertainty_service: UncertaintyCalculationService,
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

    def create_day_ahead_load_forecast(
        self, 
        province: str, 
        target_date: date,
        model_id: uuid.UUID,
        historical_days: int = 7 # 新增参数
    ) -> ForecastDTO:
        """
        用例：创建日前负荷预测。
        """
        # 1. 获取模型
        model = self._model_repo.find_by_id(model_id)
        if not model or model.forecast_type != ForecastType.LOAD:
            raise ValueError(f"ID为 {model_id} 的负荷预测模型不存在。")

        # 2. 获取包含历史和未来的数据
        # 我们需要从dataProvider获取一个足够长的序列来进行特征工程
        hist_start_date = target_date - timedelta(days=historical_days)
        
        # 假设dataProvider能处理日期范围（我们需要扩展其接口）
        combined_data = self._weather_provider.get_weather_data_for_range(
            province, hist_start_date, target_date
        )

        # 3. 调用预测引擎（它内部会处理特征工程）
        point_forecast_series = self._prediction_engine.predict(model, combined_data)
        
        # 4. 从返回的序列中只选择目标日期
        # 使用日期范围来筛选，因为预测序列包含完整的datetime索引
        target_date_start = datetime.combine(target_date, datetime.min.time())
        target_date_end = datetime.combine(target_date, datetime.max.time())
        target_forecast_series = point_forecast_series.loc[target_date_start:target_date_end]

        # 5. 识别场景并创建聚合
        # (这部分逻辑可以简化，因为天气数据已经在特征工程中被使用了)
        target_weather_data = combined_data.loc[target_date_start:target_date_end]
        weather_data_list = target_weather_data.to_dict('records')
        scenario = self._recognize_scenario(weather_data_list) 
        if not scenario:
            raise RuntimeError("无法识别有效的天气场景。")
        
        # 6. 创建并填充聚合根
        forecast_points = []
        for timestamp, value in target_forecast_series.items():
            lower, upper = self._uncertainty_service.calculate_bounds(
                predicted_value=value,
                base_uncertainty_rate=0.05,
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

        forecast = Forecast(
            province=province,
            prediction_model=model,
            matched_weather_scenario=scenario,
            time_series=forecast_points,
            creation_time=datetime.now()
        )

        # 7. 持久化聚合
        self._forecast_repo.save(forecast)

        # 8. 返回DTO，将结果与领域对象解耦
        return self._to_dto(forecast)

    def _recognize_scenario(self, weather_data: list) -> WeatherScenario | None:
        # 这是一个简化的场景识别逻辑。
        # 真实实现会更复杂，可能会调用聚类模型或复杂的规则引擎。
        # 这里我们简单地基于平均温度来判断。
        if not weather_data:
            return self._weather_scenario_repo.find_by_type(ScenarioType.MODERATE_NORMAL)

        avg_temp = sum(d['temperature'] for d in weather_data if 'temperature' in d) / len(weather_data)
        
        if avg_temp > 32:
            return self._weather_scenario_repo.find_by_type(ScenarioType.EXTREME_HOT_HUMID)
        else:
            return self._weather_scenario_repo.find_by_type(ScenarioType.MODERATE_NORMAL)

    def _prepare_model_input(self, weather_data: List[Dict[str, Any]]) -> pd.DataFrame:
        # 将原始天气数据转换为模型所需的DataFrame格式。
        # 这是一个数据处理步骤，在真实项目中会更复杂，可能包含特征工程。
        df = pd.DataFrame(weather_data)
        if 'time' not in df.columns:
            raise ValueError("天气数据中缺少 'time' 列。")
        df['time'] = pd.to_datetime(df['time'])
        return df.set_index('time')

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