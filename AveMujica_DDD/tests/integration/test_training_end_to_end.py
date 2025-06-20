#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
端到端DDD训练功能集成测试
测试完整的训练-预测流程
"""

import sys
import os
from datetime import date, datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_end_to_end_training():
    """测试端到端训练流程"""
    print("🚀 开始端到端DDD训练测试")
    print("=" * 60)
    
    try:
        # 导入必要组件
        from AveMujica_DDD.application.services.training_service import TrainingService
        from AveMujica_DDD.application.dtos.training_dto import TrainingRequestDTO
        from AveMujica_DDD.infrastructure.repositories.training_task_repository import FileTrainingTaskRepository
        from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
        
        print("✅ 1. 组件导入成功")
        
        # 创建简单的模拟模型仓储
        class MockModelRepository(IModelRepository):
            def save(self, model): pass
            def find_by_id(self, model_id): return None
            def delete(self, model_id): pass
            def find_by_type_and_region(self, model_type, region): return []
        
        # 创建依赖组件
        task_repo = FileTrainingTaskRepository("test_training_e2e")
        model_repo = MockModelRepository()
        
        # 创建训练服务（使用默认的真实实现）
        training_service = TrainingService(
            training_task_repo=task_repo,
            model_repo=model_repo
        )
        print("✅ 2. 训练服务创建成功")
        
        # 创建训练请求
        request = TrainingRequestDTO(
            model_type="convtrans",
            forecast_type="load",
            province="上海",
            train_start_date="2024-01-01",
            train_end_date="2024-01-31",
            epochs=5,  # 减少轮数用于测试
            batch_size=16,
            learning_rate=0.001
        )
        print("✅ 3. 训练请求创建成功")
        
        # 创建训练任务
        print("\n📋 创建训练任务...")
        task_id = training_service.create_training_task(request)
        print(f"✅ 4. 训练任务创建成功: {task_id}")
        
        # 检查任务状态
        print("\n📊 检查任务状态...")
        status = training_service.get_training_status(task_id)
        print(f"✅ 5. 任务状态查询成功: {status.status}")
        
        # 执行训练
        print("\n🔧 开始执行训练...")
        print("注意: 这是演示模式，使用模拟数据")
        
        try:
            training_result = training_service.execute_training(task_id)
            print(f"✅ 6. 训练执行完成!")
            print(f"   - 任务ID: {training_result.task_id}")
            print(f"   - 最终状态: {training_result.status}")
            print(f"   - 模型路径: {training_result.model_path}")
            
            # 显示训练指标
            print(f"   - 训练指标:")
            metrics = {
                "MAE": training_result.mae,
                "RMSE": training_result.rmse,
                "MAPE": training_result.mape,
                "最终损失": training_result.final_loss,
                "最佳验证损失": training_result.best_val_loss
            }
            for key, value in metrics.items():
                if value is not None:
                    print(f"     * {key}: {value}")
        
        except Exception as e:
            print(f"❌ 训练执行失败: {e}")
            print("这可能是因为缺少数据文件或依赖组件")
            return False
        
        # 测试训练历史
        print("\n📚 查询训练历史...")
        history = training_service.list_training_history(limit=5)
        print(f"✅ 7. 训练历史查询成功:")
        print(f"   - 总任务数: {history.total_tasks}")
        print(f"   - 完成任务数: {history.completed_tasks}")
        print(f"   - 失败任务数: {history.failed_tasks}")
        print(f"   - 运行中任务数: {history.running_tasks}")
        print(f"   - 最近任务数: {len(history.recent_tasks)}")
        
        # 清理测试文件
        print("\n🧹 清理测试文件...")
        import shutil
        if os.path.exists("test_training_e2e"):
            shutil.rmtree("test_training_e2e")
        print("✅ 8. 测试文件清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_training():
    """测试批量训练功能"""
    print("\n" + "=" * 60)
    print("🏭 测试批量训练功能")
    print("=" * 60)
    
    try:
        from AveMujica_DDD.application.services.training_service import TrainingService
        from AveMujica_DDD.infrastructure.repositories.training_task_repository import FileTrainingTaskRepository
        from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
        
        # 创建简单的模拟模型仓储
        class MockModelRepository(IModelRepository):
            def save(self, model): pass
            def find_by_id(self, model_id): return None
            def delete(self, model_id): pass
            def find_by_type_and_region(self, model_type, region): return []
        
        # 创建服务
        task_repo = FileTrainingTaskRepository("test_batch_training")
        model_repo = MockModelRepository()
        training_service = TrainingService(task_repo, model_repo)
        
        print("✅ 批量训练服务准备完成")
        
        # 执行批量训练
        print("\n🔧 开始批量训练...")
        task_ids = training_service.train_all_types_for_province(
            province="浙江",
            train_start_date="2024-01-01", 
            train_end_date="2024-01-15"
        )
        
        print(f"✅ 批量训练任务创建成功:")
        print(f"   - 创建了 {len(task_ids)} 个训练任务")
        for i, task_id in enumerate(task_ids, 1):
            print(f"   - 任务{i}: {task_id}")
        
        # 清理测试文件
        import shutil
        if os.path.exists("test_batch_training"):
            shutil.rmtree("test_batch_training")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """测试训练组件基础功能"""
    print("\n" + "=" * 60)
    print("🔧 测试训练组件")
    print("=" * 60)
    
    try:
        # 测试训练引擎导入
        from AveMujica_DDD.infrastructure.adapters.real_training_engine import RealTrainingEngine
        from AveMujica_DDD.infrastructure.adapters.real_data_preprocessor import RealDataPreprocessor
        from AveMujica_DDD.infrastructure.adapters.real_model_persistence import RealModelPersistence
        
        print("✅ 训练组件导入成功")
        
        # 创建组件实例
        training_engine = RealTrainingEngine()
        data_preprocessor = RealDataPreprocessor()
        model_persistence = RealModelPersistence()
        
        print("✅ 训练组件实例化成功")
        
        # 测试基本功能
        assert training_engine.supports_model_type("convtrans")
        print("✅ 训练引擎功能正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始DDD端到端训练功能测试")
    print(f"📁 工作目录: {os.getcwd()}")
    print(f"🐍 Python版本: {sys.version}")
    
    success_count = 0
    total_tests = 3
    
    # 测试1: 训练组件
    print("\n" + "🔧" * 20 + " 测试训练组件 " + "🔧" * 20)
    if test_training_components():
        success_count += 1
        print("✅ 训练组件测试通过")
    else:
        print("❌ 训练组件测试失败")
    
    # 测试2: 端到端训练
    print("\n" + "🚀" * 20 + " 端到端训练测试 " + "🚀" * 20)
    if test_end_to_end_training():
        success_count += 1
        print("✅ 端到端训练测试通过")
    else:
        print("❌ 端到端训练测试失败")
    
    # 测试3: 批量训练
    print("\n" + "🏭" * 20 + " 批量训练测试 " + "🏭" * 20)
    if test_batch_training():
        success_count += 1
        print("✅ 批量训练测试通过")
    else:
        print("❌ 批量训练测试失败")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"🎯 测试总结: {success_count}/{total_tests} 个测试通过")
    
    if success_count == total_tests:
        print("🎉 所有端到端测试都通过了！")
        print("✅ DDD训练功能完全正常，可以投入使用")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 