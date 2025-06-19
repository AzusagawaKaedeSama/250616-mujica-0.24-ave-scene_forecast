import unittest
import uuid
from datetime import date, datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import sys
import os

# This is the robust way to ensure the test runner can find the project's root
# It adds the parent directory of 'AveMujica_DDD' to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from AveMujica_DDD.domain.aggregates.forecast import Forecast
from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository
from AveMujica_DDD.domain.services.uncertainty_calculation_service import UncertaintyCalculationService
from AveMujica_DDD.application.ports.i_prediction_engine import IPredictionEngine
from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider
from AveMujica_DDD.application.services.forecast_service import ForecastService


# 1. Create Fake/Mock Implementations for all dependencies
# These fakes simulate the behavior of the infrastructure and domain layers.

class FakeForecastRepository(IForecastRepository):
    def __init__(self):
        self._forecasts: Dict[uuid.UUID, Forecast] = {}
        self.save_called = False
    
    def save(self, forecast: Forecast) -> None:
        self._forecasts[forecast.forecast_id] = forecast
        self.save_called = True

    def find_by_id(self, forecast_id: uuid.UUID) -> Optional[Forecast]:
        return self._forecasts.get(forecast_id)

class FakeModelRepository(IModelRepository):
    def __init__(self):
        self.model = PredictionModel(
            name="test_load_model",
            version="1.0",
            forecast_type=ForecastType.LOAD,
            model_id=uuid.uuid4()
        )

    def find_by_id(self, model_id: uuid.UUID) -> Optional[PredictionModel]:
        return self.model if self.model.model_id == model_id else None
    
    def save(self, model: PredictionModel) -> None:
        self.model = model

class FakeWeatherScenarioRepository(IWeatherScenarioRepository):
    def find_by_type(self, scenario_type: ScenarioType) -> Optional[WeatherScenario]:
        return WeatherScenario(scenario_type, f"fake {scenario_type.value} scenario", 1.0)
    
    def save(self, scenario: WeatherScenario) -> None:
        pass # Not needed for this test

    def list_all(self) -> List[WeatherScenario]:
        return [WeatherScenario(st, "fake", 1.0) for st in ScenarioType]

class FakeWeatherDataProvider(IWeatherDataProvider):
    def get_weather_data_for_date(self, province: str, target_date: date) -> List[Dict[str, Any]]:
        return [
            {'time': datetime(2024, 7, 10, i), 'temperature': 25.0} for i in range(24)
        ]

class FakePredictionEngine(IPredictionEngine):
    def predict(self, model: PredictionModel, input_data: pd.DataFrame) -> pd.Series:
        return pd.Series([10000.0] * len(input_data), index=input_data.index)


# 2. Write the Test Case

class TestForecastService(unittest.TestCase):
    
    def setUp(self):
        """This method runs before each test."""
        # Arrange: Create instances of all our fakes
        self.fake_forecast_repo = FakeForecastRepository()
        self.fake_model_repo = FakeModelRepository()
        self.fake_weather_repo = FakeWeatherScenarioRepository()
        self.fake_weather_provider = FakeWeatherDataProvider()
        self.fake_engine = FakePredictionEngine()
        
        # Use the real UncertaintyCalculationService as it has no dependencies
        self.uncertainty_service = UncertaintyCalculationService()

        # Instantiate the service we want to test with the fakes
        self.forecast_service = ForecastService(
            forecast_repo=self.fake_forecast_repo,
            model_repo=self.fake_model_repo,
            weather_scenario_repo=self.fake_weather_repo,
            weather_provider=self.fake_weather_provider,
            prediction_engine=self.fake_engine,
            uncertainty_service=self.uncertainty_service,
        )

    def test_create_day_ahead_load_forecast_success(self):
        """
        Tests the successful creation of a day-ahead forecast.
        """
        # Arrange
        province = "上海"
        target_date = date(2024, 7, 10)
        model_id = self.fake_model_repo.model.model_id

        # Act: Call the method we are testing
        result_dto = self.forecast_service.create_day_ahead_load_forecast(
            province=province,
            target_date=target_date,
            model_id=model_id,
        )

        # Assert: Check if the outcome is what we expect
        self.assertIsNotNone(result_dto)
        self.assertEqual(result_dto.province, province)
        self.assertEqual(len(result_dto.time_series), 24)
        self.assertEqual(result_dto.time_series[0].value, 10000.0)
        
        # Check that side-effects happened (i.e., the result was saved)
        self.assertTrue(self.fake_forecast_repo.save_called)
        
        # Check if the saved forecast in the fake repo is correct
        saved_forecast = self.fake_forecast_repo.find_by_id(result_dto.forecast_id)
        self.assertIsNotNone(saved_forecast)
        if saved_forecast:
            self.assertEqual(saved_forecast.province, province)

# This allows running the test directly from the command line
if __name__ == '__main__':
    # To run all tests, navigate to the project root directory (the one containing AveMujica_DDD)
    # and run the command:
    # python -m unittest discover .
    unittest.main() 