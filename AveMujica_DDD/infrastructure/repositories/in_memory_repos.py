import uuid
from typing import Dict, List

from AveMujica_DDD.domain.aggregates.forecast import Forecast
from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository


class InMemoryForecastRepository(IForecastRepository):
    """内存中的预测仓储实现，用于测试。"""
    def __init__(self):
        self._forecasts: Dict[uuid.UUID, Forecast] = {}

    def save(self, forecast: Forecast):
        self._forecasts[forecast.forecast_id] = forecast
        print(f"Forecast {forecast.forecast_id} saved in-memory.")

    def find_by_id(self, forecast_id: uuid.UUID) -> Forecast | None:
        return self._forecasts.get(forecast_id)


class InMemoryModelRepository(IModelRepository):
    """内存中的模型仓储实现，用于测试。"""
    def __init__(self):
        self._models: Dict[uuid.UUID, PredictionModel] = {}

    def save(self, model: PredictionModel):
        self._models[model.model_id] = model

    def find_by_id(self, model_id: uuid.UUID) -> PredictionModel | None:
        return self._models.get(model_id)

    def seed_dummy_model(self, model_id: uuid.UUID, name: str):
        """辅助方法，用于快速添加一个测试模型。"""
        if model_id not in self._models:
            dummy_model = PredictionModel(
                model_id=model_id,
                name=name,
                version="1.0",
                forecast_type=ForecastType.LOAD,
                file_path=f"models/dummy/{name}.pkl",
                target_column="load",
                feature_columns=["temperature", "humidity", "dayofweek"]
            )
            self.save(dummy_model)
            print(f"Dummy model {name} seeded in-memory.")


class InMemoryWeatherScenarioRepository(IWeatherScenarioRepository):
    """内存中的天气场景仓储实现，用于测试。"""
    def __init__(self):
        self._scenarios: Dict[ScenarioType, WeatherScenario] = {}
        self._seed_default_scenarios()

    def save(self, scenario: WeatherScenario):
        """保存一个天气场景。"""
        self._scenarios[scenario.scenario_type] = scenario
        print(f"Weather scenario for {scenario.scenario_type.value} saved/updated in-memory.")

    def find_by_type(self, scenario_type: ScenarioType) -> WeatherScenario | None:
        return self._scenarios.get(scenario_type)

    def list_all(self) -> List[WeatherScenario]:
        """列出所有天气场景。"""
        return list(self._scenarios.values())

    def _seed_default_scenarios(self):
        """创建并存储默认的场景实例。"""
        self.save(WeatherScenario(
            scenario_type=ScenarioType.MODERATE_NORMAL,
            description="温和正常天气",
            uncertainty_multiplier=1.0
        ))
        self.save(WeatherScenario(
            scenario_type=ScenarioType.EXTREME_HOT_HUMID,
            description="极端高温高湿",
            uncertainty_multiplier=3.0
        ))
        print("Default weather scenarios seeded in-memory.") 