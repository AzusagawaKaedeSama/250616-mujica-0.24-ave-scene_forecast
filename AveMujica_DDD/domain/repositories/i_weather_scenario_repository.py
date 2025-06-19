from abc import ABC, abstractmethod
from typing import List, Optional

from ..aggregates.weather_scenario import WeatherScenario, ScenarioType

class IWeatherScenarioRepository(ABC):
    """
    天气场景(WeatherScenario)聚合的仓储接口。
    """

    @abstractmethod
    def save(self, scenario: WeatherScenario) -> None:
        """
        保存一个WeatherScenario聚合实例。
        如果已存在同类型的场景，则更新它。
        """
        raise NotImplementedError

    @abstractmethod
    def find_by_type(self, scenario_type: ScenarioType) -> Optional[WeatherScenario]:
        """
        根据场景类型查找天气场景。
        """
        raise NotImplementedError
    
    @abstractmethod
    def list_all(self) -> List[WeatherScenario]:
        """
        列出所有可用的天气场景。
        """
        raise NotImplementedError 