import uuid
from typing import Dict, List

from AveMujica_DDD.domain.aggregates.forecast import Forecast
from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
from AveMujica_DDD.domain.aggregates.training_task import TrainingTask, TrainingStatus, ModelType
from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository
from AveMujica_DDD.domain.repositories.i_training_task_repository import ITrainingTaskRepository


class InMemoryForecastRepository(IForecastRepository):
    """内存中的预测仓储实现，用于测试。"""
    def __init__(self):
        self._forecasts: Dict[uuid.UUID, Forecast] = {}

    def save(self, forecast: Forecast):
        self._forecasts[forecast.forecast_id] = forecast

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

    def find_by_type_and_region(self, forecast_type: ForecastType, region: str) -> List[PredictionModel]:
        """根据预测类型和区域查找模型"""
        return [
            model for model in self._models.values() 
            if model.forecast_type == forecast_type and region in model.name
        ]
    
    def list_all(self) -> List[PredictionModel]:
        """列出所有模型"""
        return list(self._models.values())

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



class InMemoryWeatherScenarioRepository(IWeatherScenarioRepository):
    """内存中的天气场景仓储实现，用于测试。"""
    def __init__(self):
        self._scenarios: Dict[ScenarioType, WeatherScenario] = {}
        self._seed_default_scenarios()

    def save(self, scenario: WeatherScenario):
        """保存一个天气场景。"""
        self._scenarios[scenario.scenario_type] = scenario

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
            uncertainty_multiplier=1.0,
            typical_features={"temperature": 20.0, "humidity": 60.0, "wind_speed": 4.0, "precipitation": 0.0},
            power_system_impact="系统平稳运行，基准不确定性",
            operation_suggestions="标准运行模式，常规备用容量配置"
        ))
        self.save(WeatherScenario(
            scenario_type=ScenarioType.EXTREME_HOT_HUMID,
            description="极端高温高湿",
            uncertainty_multiplier=3.0,
            typical_features={"temperature": 35.0, "humidity": 85.0, "wind_speed": 3.0, "precipitation": 0.0},
            power_system_impact="空调负荷极高，电网压力巨大，可能出现负荷激增",
            operation_suggestions="全力运行发电机组，准备需求响应措施，监控线路温度"
        ))


class InMemoryTrainingTaskRepository(ITrainingTaskRepository):
    """内存中的训练任务仓储实现，用于测试。"""
    
    def __init__(self):
        self._tasks: Dict[str, TrainingTask] = {}
    
    def save(self, task: TrainingTask) -> None:
        """保存训练任务"""
        self._tasks[task.task_id] = task
        
    def find_by_id(self, task_id: str) -> TrainingTask | None:
        """根据ID查找训练任务"""
        return self._tasks.get(task_id)
        
    def find_recent_tasks(self, limit: int = 10) -> List[TrainingTask]:
        """查找最近的训练任务"""
        sorted_tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        return sorted_tasks[:limit]
        
    def delete(self, task_id: str) -> bool:
        """删除训练任务"""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False
    
    def find_by_status(self, status: TrainingStatus) -> List[TrainingTask]:
        """根据状态查找训练任务"""
        return [task for task in self._tasks.values() if task.status == status]
    
    def find_by_province_and_type(self, 
                                  province: str, 
                                  model_type: ModelType,
                                  forecast_type: ForecastType) -> List[TrainingTask]:
        """根据省份、模型类型和预测类型查找训练任务"""
        return [
            task for task in self._tasks.values() 
            if (task.province == province and 
                task.model_type == model_type and 
                task.forecast_type == forecast_type)
        ]
    
    def find_running_tasks(self) -> List[TrainingTask]:
        """查找正在运行的训练任务"""
        return self.find_by_status(TrainingStatus.RUNNING) 