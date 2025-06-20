"""
训练任务仓储的基础设施实现
"""

import os
import json
from typing import List, Optional
from datetime import datetime, date
from ...domain.repositories.i_training_task_repository import ITrainingTaskRepository
from ...domain.aggregates.training_task import TrainingTask, TrainingStatus, ModelType, TrainingConfig
from ...domain.aggregates.prediction_model import ForecastType


class FileTrainingTaskRepository(ITrainingTaskRepository):
    """基于文件系统的训练任务仓储实现"""
    
    def __init__(self, base_path: str = "data/training_tasks"):
        self._base_path = base_path
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """确保存储目录存在"""
        os.makedirs(self._base_path, exist_ok=True)
    
    def _get_task_file_path(self, task_id: str) -> str:
        """获取任务文件路径"""
        return os.path.join(self._base_path, f"{task_id}.json")
    
    def save(self, task: TrainingTask) -> None:
        """保存训练任务"""
        file_path = self._get_task_file_path(task.task_id)
        
        # 将任务转换为字典并保存
        task_data = task.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2)
    
    def find_by_id(self, task_id: str) -> Optional[TrainingTask]:
        """根据ID查找训练任务"""
        file_path = self._get_task_file_path(task_id)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            return self._dict_to_task(task_data)
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
            return None
    
    def find_by_status(self, status: TrainingStatus) -> List[TrainingTask]:
        """根据状态查找训练任务"""
        tasks = []
        
        for filename in os.listdir(self._base_path):
            if filename.endswith('.json'):
                task_id = filename[:-5]  # 移除.json后缀
                task = self.find_by_id(task_id)
                if task and task.status == status:
                    tasks.append(task)
        
        return tasks
    
    def find_by_province_and_type(self, 
                                  province: str, 
                                  model_type: ModelType,
                                  forecast_type: ForecastType) -> List[TrainingTask]:
        """根据省份、模型类型和预测类型查找训练任务"""
        tasks = []
        
        for filename in os.listdir(self._base_path):
            if filename.endswith('.json'):
                task_id = filename[:-5]
                task = self.find_by_id(task_id)
                if (task and 
                    task.province == province and 
                    task.model_type == model_type and 
                    task.forecast_type == forecast_type):
                    tasks.append(task)
        
        return tasks
    
    def find_running_tasks(self) -> List[TrainingTask]:
        """查找正在运行的训练任务"""
        return self.find_by_status(TrainingStatus.RUNNING)
    
    def find_recent_tasks(self, limit: int = 10) -> List[TrainingTask]:
        """查找最近的训练任务"""
        tasks = []
        
        # 获取所有任务文件
        files = [f for f in os.listdir(self._base_path) if f.endswith('.json')]
        
        # 按修改时间排序
        files.sort(key=lambda f: os.path.getmtime(os.path.join(self._base_path, f)), reverse=True)
        
        # 加载前limit个任务
        for filename in files[:limit]:
            task_id = filename[:-5]
            task = self.find_by_id(task_id)
            if task:
                tasks.append(task)
        
        return tasks
    
    def delete(self, task_id: str) -> bool:
        """删除训练任务"""
        file_path = self._get_task_file_path(task_id)
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting task {task_id}: {e}")
            return False
    
    def _dict_to_task(self, task_data: dict) -> TrainingTask:
        """将字典转换为训练任务对象"""
        
        # 重建配置对象
        config = TrainingConfig(
            seq_length=task_data['config'].get('seq_length', 96),
            pred_length=task_data['config'].get('pred_length', 1),
            batch_size=task_data['config'].get('batch_size', 32),
            learning_rate=task_data['config'].get('lr', 1e-4),
            epochs=task_data['config'].get('epochs', 50),
            patience=task_data['config'].get('patience', 10),
            peak_hours=tuple(task_data['config'].get('peak_hours', (8, 20))),
            valley_hours=tuple(task_data['config'].get('valley_hours', (0, 6))),
            peak_weight=task_data['config'].get('peak_weight', 2.5),
            valley_weight=task_data['config'].get('valley_weight', 1.5),
            weather_features=task_data['config'].get('weather_features'),
            quantiles=task_data['config'].get('quantiles')
        )
        
        # 重建任务对象
        task = TrainingTask(
            task_id=task_data['task_id'],
            model_type=ModelType(task_data['model_type']),
            forecast_type=ForecastType(task_data['forecast_type']),
            province=task_data['province'],
            train_start_date=date.fromisoformat(task_data['train_start_date']),
            train_end_date=date.fromisoformat(task_data['train_end_date']),
            config=config
        )
        
        # 设置状态和时间
        task.status = TrainingStatus(task_data['status'])
        task.created_at = datetime.fromisoformat(task_data['created_at'])
        
        if task_data.get('started_at'):
            task.started_at = datetime.fromisoformat(task_data['started_at'])
        if task_data.get('completed_at'):
            task.completed_at = datetime.fromisoformat(task_data['completed_at'])
        
        task.error_message = task_data.get('error_message')
        task.model_path = task_data.get('model_path')
        task.logs = task_data.get('logs', [])
        
        # 恢复指标
        metrics_data = task_data.get('metrics', {})
        task.metrics.final_loss = metrics_data.get('final_loss')
        task.metrics.best_val_loss = metrics_data.get('best_val_loss')
        task.metrics.epochs_trained = metrics_data.get('epochs_trained')
        task.metrics.mae = metrics_data.get('mae')
        task.metrics.rmse = metrics_data.get('rmse')
        task.metrics.mape = metrics_data.get('mape')
        
        return task


class InMemoryTrainingTaskRepository(ITrainingTaskRepository):
    """内存实现的训练任务仓储（用于测试）"""
    
    def __init__(self):
        self._tasks = {}
    
    def save(self, task: TrainingTask) -> None:
        """保存训练任务"""
        self._tasks[task.task_id] = task
    
    def find_by_id(self, task_id: str) -> Optional[TrainingTask]:
        """根据ID查找训练任务"""
        return self._tasks.get(task_id)
    
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
    
    def find_recent_tasks(self, limit: int = 10) -> List[TrainingTask]:
        """查找最近的训练任务"""
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]
    
    def delete(self, task_id: str) -> bool:
        """删除训练任务"""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False 