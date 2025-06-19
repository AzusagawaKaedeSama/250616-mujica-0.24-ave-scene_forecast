from ...domain.aggregates.prediction_model import PredictionModel, ForecastType
from ...domain.repositories.i_model_repository import IModelRepository
from ..dtos.model_training_dto import ModelTrainingDTO
from ..ports.i_feature_engineering_service import IFeatureEngineeringService
from ..ports.i_historical_data_provider import IHistoricalDataProvider
from ..ports.i_model_trainer import IModelTrainer

class TrainingService:
    """
    应用服务：负责编排和执行模型训练的用例。
    """
    def __init__(
        self,
        historical_provider: IHistoricalDataProvider,
        feature_service: IFeatureEngineeringService,
        model_trainer: IModelTrainer,
        model_repo: IModelRepository,
    ):
        self._historical_provider = historical_provider
        self._feature_service = feature_service
        self._model_trainer = model_trainer
        self._model_repo = model_repo

    def train_new_model(
        self,
        province: str,
        model_name: str,
        model_version: str,
        forecast_type: str, # e.g., "LOAD", "PV"
    ) -> ModelTrainingDTO:
        """
        用例：训练一个新模型并将其注册到系统中。
        """
        # 1. 获取数据
        historical_data = self._historical_provider.get_historical_data(province)

        # 2. 特征工程
        features, labels = self._feature_service.preprocess_for_training(historical_data)

        # 3. 调用外部引擎进行训练
        training_results = self._model_trainer.train(features, labels)

        # 4. 创建领域对象
        model = PredictionModel(
            name=model_name,
            version=model_version,
            forecast_type=ForecastType(forecast_type.upper()),
            description=f"为 {province} 训练的模型，基于 {len(historical_data)} 条数据。",
            # 可以在这里添加更多元数据，例如 training_results['performance_metrics']
        )
        
        # 5. 持久化领域对象
        self._model_repo.save(model)

        # 6. 返回DTO
        return ModelTrainingDTO(
            model_id=model.model_id,
            model_name=model.name,
            model_version=model.version,
            message=f"模型 {model.name} v{model.version} 训练成功。",
            performance_metrics=training_results.get('performance_metrics', {})
        ) 