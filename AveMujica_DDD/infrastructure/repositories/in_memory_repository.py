import uuid
from typing import Dict, List, Optional

from ...domain.aggregates.forecast import Forecast
from ...domain.aggregates.prediction_model import PredictionModel, ForecastType
from ...domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
from ...domain.repositories.i_forecast_repository import IForecastRepository
from ...domain.repositories.i_model_repository import IModelRepository
from ...domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository

# --- In-Memory Implementations for Repositories ---

class InMemoryForecastRepository(IForecastRepository):
    def __init__(self):
        self._forecasts: Dict[uuid.UUID, Forecast] = {}

    def save(self, forecast: Forecast) -> None:
        print(f"--- [InMemoryRepo] 正在保存预测 {forecast.forecast_id} ---")
        self._forecasts[forecast.forecast_id] = forecast

    def find_by_id(self, forecast_id: uuid.UUID) -> Optional[Forecast]:
        return self._forecasts.get(forecast_id)

class InMemoryModelRepository(IModelRepository):
    def __init__(self):
        self._models: Dict[uuid.UUID, PredictionModel] = {}
        # Pre-seed with a default model for testing
        default_model = PredictionModel(
            name="default_load_model",
            version="1.0",
            forecast_type=ForecastType.LOAD,
            file_path="models/convtrans_weather/load/上海/best_model.pth",
            description="一个默认的、用于演示的负荷预测模型"
        )
        self.save(default_model)

    def save(self, model: PredictionModel) -> None:
        print(f"--- [InMemoryRepo] 正在保存模型 {model.model_id} ({model.name}) ---")
        self._models[model.model_id] = model

    def find_by_id(self, model_id: uuid.UUID) -> Optional[PredictionModel]:
        return self._models.get(model_id)

    def find_by_name_and_version(self, name: str, version: str) -> Optional[PredictionModel]:
        for model in self._models.values():
            if model.name == name and model.version == version:
                return model
        return None

class InMemoryWeatherScenarioRepository(IWeatherScenarioRepository):
    def __init__(self):
        self._scenarios: Dict[ScenarioType, WeatherScenario] = {}
        # Pre-seed with all scenarios
        for st in ScenarioType:
            scenario = WeatherScenario(
                scenario_type=st,
                description=f"这是一个 {st.value} 的场景。",
                uncertainty_multiplier=1.5 if "EXTREME" in st.name else 1.0
            )
            self.save(scenario)

    def save(self, scenario: WeatherScenario) -> None:
        self._scenarios[scenario.scenario_type] = scenario

    def find_by_type(self, scenario_type: ScenarioType) -> Optional[WeatherScenario]:
        return self._scenarios.get(scenario_type)

    def list_all(self) -> List[WeatherScenario]:
        return list(self._scenarios.values()) 