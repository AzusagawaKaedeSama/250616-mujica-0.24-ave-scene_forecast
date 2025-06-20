from datetime import date
from typing import Dict, Any
from ...domain.aggregates.prediction_model import PredictionModel, ForecastType
from ...domain.repositories.i_model_repository import IModelRepository
from ..dtos.model_training_dto import ModelTrainingDTO
from ..ports.i_weather_data_provider import IWeatherDataProvider
from ..ports.i_prediction_engine import IPredictionEngine


class TrainingService:
    """
    应用服务：负责编排和执行模型训练的用例。
    """
    def __init__(
        self,
        model_repo: IModelRepository,
        weather_provider: IWeatherDataProvider,
        prediction_engine: IPredictionEngine,
    ):
        self._model_repo = model_repo
        self._weather_provider = weather_provider
        self._prediction_engine = prediction_engine
        self._training_tasks = {}  # 简单的内存存储训练任务状态

    def train_model(
        self,
        province: str,
        forecast_type: str,
        start_date: date,
        end_date: date,
        task_id: str
    ) -> Dict[str, Any]:
        """
        用例：训练一个新模型。
        """
        try:
            # 更新任务状态
            self._training_tasks[task_id] = {
                "status": "running",
                "progress": 0,
                "message": "Starting training..."
            }
            
            # 1. 获取数据（在真实环境中这里会加载实际数据）
            print(f"Training model for {province} {forecast_type} from {start_date} to {end_date}")
            
            # 模拟训练过程
            self._training_tasks[task_id].update({
                "progress": 50,
                "message": "Training in progress..."
            })
            
            # 2. 创建模型对象
            model_name = f"{province}_{forecast_type}_model"
            model = PredictionModel(
                name=model_name,
                version="1.0",
                forecast_type=ForecastType(forecast_type.upper() if forecast_type.upper() in ['LOAD', 'PV', 'WIND'] else 'LOAD'),
                description=f"为 {province} 训练的{forecast_type}模型，训练期间：{start_date} 到 {end_date}",
                file_path=f"models/{province}_{forecast_type}_model.pt"
            )
            
            # 3. 保存模型
            self._model_repo.save(model)
            
            # 4. 更新任务状态为完成
            self._training_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "message": f"Training completed successfully for {model_name}"
            })
            
            return {
                "model_id": str(model.model_id),
                "model_name": model.name,
                "status": "completed",
                "message": f"模型 {model.name} 训练成功"
            }
            
        except Exception as e:
            # 更新任务状态为失败
            self._training_tasks[task_id] = {
                "status": "failed",
                "progress": 0,
                "message": f"Training failed: {str(e)}"
            }
            raise

    def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取训练任务状态。
        """
        return self._training_tasks.get(task_id, {
            "status": "not_found",
            "progress": 0,
            "message": f"Task {task_id} not found"
        })

    def train_new_model(
        self,
        province: str,
        model_name: str,
        model_version: str,
        forecast_type: str, # e.g., "LOAD", "PV"
    ) -> ModelTrainingDTO:
        """
        用例：训练一个新模型并将其注册到系统中（保留原有方法用于兼容性）。
        """
        # 创建领域对象
        model = PredictionModel(
            name=model_name,
            version=model_version,
            forecast_type=ForecastType(forecast_type.upper()),
            description=f"为 {province} 训练的模型",
        )
        
        # 持久化领域对象
        self._model_repo.save(model)

        # 返回DTO
        return ModelTrainingDTO(
            model_id=model.model_id,
            model_name=model.name,
            model_version=model.version,
            message=f"模型 {model.name} v{model.version} 训练成功。",
            performance_metrics={}
        ) 