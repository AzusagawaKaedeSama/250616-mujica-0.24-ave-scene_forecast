"""
训练应用服务 - DDD架构的训练功能核心
整合原mujica-0.24版本的所有训练能力
"""

import os
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from ..dtos.training_dto import (
    TrainingRequestDTO, 
    TrainingResultDTO, 
    TrainingProgressDTO, 
    TrainingHistoryDTO
)
from ..ports.i_training_engine import ITrainingEngine, IDataPreprocessor, IModelPersistence
from ...domain.aggregates.training_task import TrainingTask, TrainingConfig, ModelType, TrainingStatus
from ...domain.repositories.i_training_task_repository import ITrainingTaskRepository
from ...domain.repositories.i_model_repository import IModelRepository


class TrainingService:
    """训练应用服务 - 协调训练任务的完整生命周期"""
    
    def __init__(self,
                 training_task_repo: ITrainingTaskRepository,
                 model_repo: IModelRepository,
                 training_engines: Optional[Dict[str, ITrainingEngine]] = None,
                 data_preprocessor: Optional[IDataPreprocessor] = None,
                 model_persistence: Optional[IModelPersistence] = None):
        
        self._training_task_repo = training_task_repo
        self._model_repo = model_repo
        
        # 使用默认实现（如果未提供）
        if training_engines is None:
            from ...infrastructure.adapters.real_training_engine import RealTrainingEngine
            default_engine = RealTrainingEngine()
            self._training_engines = {
                'convtrans': default_engine,
                'convtrans_peak': default_engine,
                'convtrans_weather': default_engine,
                'probabilistic': default_engine,
                'interval': default_engine
            }
        else:
            self._training_engines = training_engines
            
        if data_preprocessor is None:
            from ...infrastructure.adapters.real_data_preprocessor import RealDataPreprocessor
            self._data_preprocessor = RealDataPreprocessor()
        else:
            self._data_preprocessor = data_preprocessor
            
        if model_persistence is None:
            from ...infrastructure.adapters.real_model_persistence import RealModelPersistence
            self._model_persistence = RealModelPersistence()
        else:
            self._model_persistence = model_persistence
    
    def train_model(self, 
                   province: str,
                   forecast_type: str,
                   start_date: date,
                   end_date: date,
                   task_id: str,
                   **kwargs) -> Dict[str, Any]:
        """
        API便捷方法：训练模型
        
        Args:
            province: 省份名称
            forecast_type: 预测类型 (load, pv, wind)
            start_date: 训练开始日期
            end_date: 训练结束日期
            task_id: 任务ID
            **kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        print(f"🚀 开始训练模型: {province} - {forecast_type}")
        
        try:
            # 创建训练请求DTO
            request = TrainingRequestDTO(
                model_type=kwargs.get('model_type', 'convtrans'),
                forecast_type=forecast_type,
                province=province,
                train_start_date=start_date.isoformat(),
                train_end_date=end_date.isoformat(),
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32),
                learning_rate=kwargs.get('learning_rate', 0.001),
                retrain=kwargs.get('retrain', True),
                weather_features=kwargs.get('weather_features')
            )
            
            # 使用提供的task_id创建训练任务
            print(f"🔧 创建训练任务: {task_id}")
            
            # 验证请求参数
            self._validate_training_request(request)
            
            # 创建训练配置
            config = TrainingConfig(
                seq_length=96,  # 24小时，15分钟间隔
                pred_length=1,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                epochs=request.epochs,
                patience=request.patience,
                peak_hours=request.peak_hours,
                valley_hours=request.valley_hours,
                peak_weight=request.peak_weight,
                valley_weight=request.valley_weight,
                weather_features=request.weather_features,
                quantiles=request.quantiles
            )
            
            # 创建训练任务聚合，使用提供的task_id
            task = TrainingTask(
                model_type=request.to_model_type(),
                forecast_type=request.to_forecast_type(),
                province=request.province,
                train_start_date=date.fromisoformat(request.train_start_date),
                train_end_date=date.fromisoformat(request.train_end_date),
                config=config,
                retrain=request.retrain,
                task_id=task_id  # 使用提供的task_id
            )
            
            # 保存任务
            self._training_task_repo.save(task)
            print(f"✅ 训练任务已创建: {task.task_id}")
            
            # 直接执行训练
            result_dto = self.execute_training(task_id)
            
            # 转换为字典格式返回
            return {
                "task_id": result_dto.task_id,
                "status": result_dto.status,
                "model_type": result_dto.model_type,
                "forecast_type": result_dto.forecast_type,
                "province": result_dto.province,
                "created_at": result_dto.created_at,
                "started_at": result_dto.started_at,
                "completed_at": result_dto.completed_at,
                "model_path": result_dto.model_path,
                "final_loss": result_dto.final_loss,
                "mae": result_dto.mae,
                "rmse": result_dto.rmse,
                "mape": result_dto.mape,
                "logs": result_dto.logs
            }
            
        except Exception as e:
            print(f"❌ 训练模型失败: {str(e)}")
            # 返回错误信息
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "province": province,
                "forecast_type": forecast_type
            }
    
    def create_training_task(self, request: TrainingRequestDTO) -> str:
        """
        创建训练任务
        
        Args:
            request: 训练请求DTO
            
        Returns:
            任务ID
        """
        print(f"🔧 创建训练任务: {request.model_type} - {request.forecast_type} - {request.province}")
        
        # 验证请求参数
        self._validate_training_request(request)
        
        # 创建训练配置
        config = TrainingConfig(
            seq_length=96,  # 24小时，15分钟间隔
            pred_length=1,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            epochs=request.epochs,
            patience=request.patience,
            peak_hours=request.peak_hours,
            valley_hours=request.valley_hours,
            peak_weight=request.peak_weight,
            valley_weight=request.valley_weight,
            weather_features=request.weather_features,
            quantiles=request.quantiles
        )
        
        # 创建训练任务聚合
        task = TrainingTask(
            model_type=request.to_model_type(),
            forecast_type=request.to_forecast_type(),
            province=request.province,
            train_start_date=date.fromisoformat(request.train_start_date),
            train_end_date=date.fromisoformat(request.train_end_date),
            config=config,
            retrain=request.retrain
        )
        
        # 保存任务
        self._training_task_repo.save(task)
        
        print(f"✅ 训练任务已创建: {task.task_id}")
        return task.task_id
    
    def execute_training(self, task_id: str) -> TrainingResultDTO:
        """
        执行训练任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            训练结果DTO
        """
        print(f"🚀 开始执行训练任务: {task_id}")
        
        # 获取训练任务
        task = self._training_task_repo.find_by_id(task_id)
        if not task:
            raise ValueError(f"找不到训练任务: {task_id}")
        
        try:
            # 开始训练
            task.start_training()
            self._training_task_repo.save(task)
            
            # 选择合适的训练引擎
            engine = self._select_training_engine(task)
            
            # 准备数据路径
            data_path = self._get_data_path(task)
            print(f"📁 数据路径: {data_path}")
            
            # 执行训练流程
            final_metrics = self._execute_training_pipeline(task, engine, data_path)
            
            # 完成训练
            model_path = task.get_model_directory()
            task.complete_training(model_path, final_metrics)
            self._training_task_repo.save(task)
            
            print(f"✅ 训练任务完成: {task_id}")
            return TrainingResultDTO.from_training_task(task)
            
        except Exception as e:
            print(f"❌ 训练任务失败: {task_id} - {str(e)}")
            task.fail_training(str(e))
            self._training_task_repo.save(task)
            
            # 清理失败的训练文件
            self._model_persistence.cleanup_failed_training(task)
            
            raise e
    
    def get_training_status(self, task_id: str) -> TrainingResultDTO:
        """获取训练状态"""
        task = self._training_task_repo.find_by_id(task_id)
        if not task:
            raise ValueError(f"找不到训练任务: {task_id}")
        
        return TrainingResultDTO.from_training_task(task)
    
    def list_training_history(self, limit: int = 10) -> TrainingHistoryDTO:
        """获取训练历史"""
        # 获取最近的训练任务
        recent_tasks = self._training_task_repo.find_recent_tasks(limit)
        recent_dtos = [TrainingResultDTO.from_training_task(task) for task in recent_tasks]
        
        # 统计各状态的任务数量
        all_tasks = self._training_task_repo.find_recent_tasks(1000)  # 获取更多用于统计
        total_tasks = len(all_tasks)
        completed_tasks = len([t for t in all_tasks if t.status == TrainingStatus.COMPLETED])
        failed_tasks = len([t for t in all_tasks if t.status == TrainingStatus.FAILED])
        running_tasks = len([t for t in all_tasks if t.status == TrainingStatus.RUNNING])
        
        return TrainingHistoryDTO(
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            running_tasks=running_tasks,
            recent_tasks=recent_dtos
        )
    
    def cancel_training(self, task_id: str) -> bool:
        """取消训练任务"""
        task = self._training_task_repo.find_by_id(task_id)
        if not task:
            return False
        
        try:
            task.cancel_training()
            self._training_task_repo.save(task)
            
            # 清理训练文件
            self._model_persistence.cleanup_failed_training(task)
            
            print(f"🛑 训练任务已取消: {task_id}")
            return True
        except Exception as e:
            print(f"❌ 取消训练任务失败: {task_id} - {str(e)}")
            return False
    
    def train_all_types_for_province(self, 
                                   province: str,
                                   train_start_date: str,
                                   train_end_date: str) -> List[str]:
        """为指定省份训练所有类型的模型（类似原系统的批量训练）"""
        print(f"🏭 为 {province} 开始批量训练所有模型类型")
        
        task_ids = []
        
        # 训练基础模型类型
        model_configs = [
            # 通用模型
            {"model_type": "convtrans", "forecast_type": "load"},
            {"model_type": "convtrans", "forecast_type": "pv"},
            {"model_type": "convtrans", "forecast_type": "wind"},
            
            # 峰谷感知模型（主要用于负荷）
            {"model_type": "convtrans_peak", "forecast_type": "load"},
            
            # 天气感知模型
            {"model_type": "convtrans_weather", "forecast_type": "load"},
            {"model_type": "convtrans_weather", "forecast_type": "pv"},
            {"model_type": "convtrans_weather", "forecast_type": "wind"},
            
            # 概率预测模型
            {"model_type": "probabilistic", "forecast_type": "load", "quantiles": [0.1, 0.5, 0.9]},
            
            # 区间预测模型
            {"model_type": "interval", "forecast_type": "load"}
        ]
        
        for config in model_configs:
            try:
                request = TrainingRequestDTO(
                    model_type=config["model_type"],
                    forecast_type=config["forecast_type"],
                    province=province,
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    retrain=True,
                    quantiles=config.get("quantiles")
                )
                
                task_id = self.create_training_task(request)
                task_ids.append(task_id)
                
                print(f"✅ 已创建训练任务: {config['model_type']} - {config['forecast_type']}")
                
            except Exception as e:
                print(f"❌ 创建训练任务失败: {config} - {str(e)}")
                continue
        
        print(f"🎯 批量训练创建完成，共 {len(task_ids)} 个任务")
        return task_ids
    
    def _validate_training_request(self, request: TrainingRequestDTO) -> None:
        """验证训练请求"""
        if not request.province:
            raise ValueError("省份名称不能为空")
        
        if not request.train_start_date or not request.train_end_date:
            raise ValueError("训练日期范围不能为空")
        
        try:
            start_date = date.fromisoformat(request.train_start_date)
            end_date = date.fromisoformat(request.train_end_date)
            if start_date >= end_date:
                raise ValueError("训练开始日期必须早于结束日期")
        except ValueError as e:
            raise ValueError(f"日期格式错误: {str(e)}")
        
        # 验证模型类型
        valid_model_types = ["convtrans", "convtrans_peak", "convtrans_weather", "probabilistic", "interval"]
        if request.model_type not in valid_model_types:
            raise ValueError(f"不支持的模型类型: {request.model_type}")
        
        # 验证预测类型
        valid_forecast_types = ["load", "pv", "wind"]
        if request.forecast_type not in valid_forecast_types:
            raise ValueError(f"不支持的预测类型: {request.forecast_type}")
    
    def _select_training_engine(self, task: TrainingTask) -> ITrainingEngine:
        """选择合适的训练引擎"""
        model_type = task.model_type.value
        
        if model_type not in self._training_engines:
            raise ValueError(f"找不到适合的训练引擎: {model_type}")
        
        engine = self._training_engines[model_type]
        if not engine.supports_model_type(model_type):
            raise ValueError(f"训练引擎不支持模型类型: {model_type}")
        
        return engine
    
    def _get_data_path(self, task: TrainingTask) -> str:
        """获取数据文件路径"""
        if task.is_weather_aware():
            # 天气感知模型使用包含天气数据的文件
            data_path = f"data/timeseries_{task.forecast_type.value}_weather_{task.province}.csv"
        else:
            # 其他模型使用标准数据文件
            data_path = f"data/timeseries_{task.forecast_type.value}_{task.province}.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}")
        
        return data_path
    
    def _execute_training_pipeline(self, 
                                 task: TrainingTask, 
                                 engine: ITrainingEngine,
                                 data_path: str) -> Dict[str, Any]:
        """执行训练管道"""
        print(f"📊 开始训练管道: {task.model_type.value}")
        
        # 1. 准备数据
        task.add_log("开始数据准备")
        X_train, y_train, X_val, y_val = engine.prepare_data(task, data_path)
        
        input_shape = X_train.shape[1:]
        task.add_log(f"数据准备完成: 训练集 {X_train.shape}, 验证集 {X_val.shape}")
        
        # 2. 创建模型
        task.add_log("创建模型")
        model = engine.create_model(task, input_shape)
        task.add_log(f"模型创建完成: {task.model_type.value}")
        
        # 3. 定义进度回调
        def progress_callback(progress: TrainingProgressDTO):
            task.update_progress(progress.epoch, progress.current_loss, progress.validation_loss)
            self._training_task_repo.save(task)
        
        # 4. 训练模型
        task.add_log("开始模型训练")
        training_metrics = engine.train_model(
            task, model, X_train, y_train, X_val, y_val, progress_callback
        )
        
        # 5. 保存模型
        task.add_log("保存模型")
        model_path = engine.save_model(task, model)
        task.add_log(f"模型已保存: {model_path}")
        
        # 6. 保存训练元数据
        self._model_persistence.save_training_metadata(task, training_metrics)
        
        return training_metrics 