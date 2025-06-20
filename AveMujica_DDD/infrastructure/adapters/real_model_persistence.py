"""
真实模型持久化实现
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from ...application.ports.i_training_engine import IModelPersistence
from ...domain.aggregates.training_task import TrainingTask


class RealModelPersistence(IModelPersistence):
    """真实的模型持久化实现"""
    
    def save_model_files(self, task: TrainingTask, model: Any, additional_files: Optional[Dict[str, Any]] = None) -> str:
        """保存模型文件和相关资源"""
        save_dir = task.get_model_directory()
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        if hasattr(model, 'save'):
            model.save(save_dir=save_dir)
        
        # 保存额外文件
        if additional_files:
            for filename, content in additional_files.items():
                file_path = os.path.join(save_dir, filename)
                if isinstance(content, dict):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(content, f, ensure_ascii=False, indent=2)
                else:
                    with open(file_path, 'wb') as f:
                        f.write(content)
        
        return save_dir
    
    def save_training_metadata(self, task: TrainingTask, metrics: Dict[str, Any]) -> None:
        """保存训练元数据"""
        metadata = {
            "task_id": task.task_id,
            "model_type": task.model_type.value,
            "forecast_type": task.forecast_type.value,
            "province": task.province,
            "training_start": task.train_start_date.isoformat(),
            "training_end": task.train_end_date.isoformat(),
            "metrics": metrics,
            "config": task.config.to_dict(),
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(task.get_model_directory(), "training_metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def cleanup_failed_training(self, task: TrainingTask) -> None:
        """清理失败训练的文件"""
        import shutil
        
        model_dir = task.get_model_directory()
        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                print(f"✅ 已清理失败训练文件: {model_dir}")
            except Exception as e:
                print(f"❌ 清理文件失败: {e}") 