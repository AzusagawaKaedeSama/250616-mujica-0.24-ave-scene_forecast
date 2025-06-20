#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 MUJICA DDD 天气感知负荷预测系统启动器
优雅启动重构后的四层DDD架构系统
"""

import sys
import os
from datetime import datetime, date, timedelta
from typing import Optional
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AveMujica_DDD.api import DIContainer, create_app
from AveMujica_DDD.domain.aggregates.prediction_model import ForecastType

class MujicaDDDLauncher:
    """MUJICA DDD系统启动器 - 简洁优雅的启动体验"""
    
    def __init__(self, use_real_implementations: bool = True):
        print("🌟 初始化 MUJICA DDD 天气感知负荷预测系统...")
        self.container = DIContainer(use_real_implementations=use_real_implementations)
        print("✅ DDD容器初始化完成")

    def start_api_server(self, host: str = "0.0.0.0", port: int = 5001, debug: bool = True):
        """启动API服务器"""
        print(f"🚀 启动API服务器: http://{host}:{port}")
        app = create_app(self.container)
        app.run(host=host, port=port, debug=debug)

    def run_forecast_demo(self, province: str = "上海", days_ahead: int = 1) -> dict:
        """运行预测演示"""
        print(f"📊 开始为 {province} 进行 {days_ahead} 天的负荷预测...")
        
        start_date = date.today() + timedelta(days=1)
        end_date = start_date + timedelta(days=days_ahead - 1)
        
        print(f"预测日期范围: {start_date} 到 {end_date}")
        
        try:
            # 创建预测
            forecast_dto = self.container.forecast_service.create_day_ahead_forecast(
                province=province,
                start_date=start_date,
                end_date=end_date,
                forecast_type=ForecastType.LOAD
            )
            
            print("\n" + "="*50)
            print("🎯 预测结果")
            print("="*50)
            print(f"预测ID: {forecast_dto.forecast_id}")
            print(f"省份: {forecast_dto.province}")
            print(f"创建时间: {forecast_dto.creation_time}")
            print(f"使用模型: {forecast_dto.model_name}")
            print(f"天气场景: {forecast_dto.scenario_type}")
            
            # 显示预测数据概览
            if forecast_dto.time_series:
                print(f"\n📈 预测数据点: {len(forecast_dto.time_series)} 个")
                
                # 显示前几个数据点
                print("\n时间               | 预测值(MW)  | 下界(MW)    | 上界(MW)")
                print("-" * 60)
                for i, point in enumerate(forecast_dto.time_series[:5]):
                    time_str = point.timestamp.strftime("%Y-%m-%d %H:%M")
                    print(f"{time_str}   |   {point.value:8.1f}   |   {point.lower_bound or 0:8.1f}   |   {point.upper_bound or 0:8.1f}")
                
                if len(forecast_dto.time_series) > 5:
                    print("...")
                
                # 统计信息
                values = [p.value for p in forecast_dto.time_series]
                print(f"\n📊 统计信息:")
                print(f"   平均预测值: {sum(values)/len(values):,.1f} MW")
                print(f"   最大预测值: {max(values):,.1f} MW")
                print(f"   最小预测值: {min(values):,.1f} MW")
                
                # 计算平均区间宽度
                intervals = [p.upper_bound - p.lower_bound for p in forecast_dto.time_series 
                           if p.upper_bound and p.lower_bound]
                if intervals:
                    print(f"   平均置信区间宽度: {sum(intervals)/len(intervals):,.1f} MW")
            
            print("="*50)
            return {
                "success": True,
                "forecast": forecast_dto,
                "summary": {
                    "province": province,
                    "forecast_id": str(forecast_dto.forecast_id),
                    "model": forecast_dto.model_name,
                    "scenario": forecast_dto.scenario_type,
                    "data_points": len(forecast_dto.time_series)
                }
            }
            
        except Exception as e:
            error_msg = f"❌ 预测失败: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def run_scenario_analysis_demo(self, province: str = "上海"):
        """运行场景分析演示"""
        print(f"🌤️ 开始为 {province} 进行天气场景分析...")
        
        try:
            # 简化的场景分析演示
            from AveMujica_DDD.domain.services.weather_scenario_recognition_service import WeatherFeatures
            
            # 模拟当前天气
            current_weather = WeatherFeatures(
                temperature=25.0,
                humidity=70.0,
                wind_speed=4.0,
                precipitation=0.0
            )
            
            # 简化演示 - 直接创建场景识别服务
            from AveMujica_DDD.domain.services.weather_scenario_recognition_service import WeatherScenarioRecognitionService
            recognition_service = WeatherScenarioRecognitionService()
            scenario_result = recognition_service.recognize_scenario(current_weather)
            
            print("\n" + "="*50)
            print("🎯 场景识别结果")
            print("="*50)
            print(f"识别场景: {scenario_result.matched_scenario.scenario_type.value}")
            print(f"相似度: {scenario_result.similarity_score:.1%}")
            print(f"置信度: {scenario_result.confidence_level}")
            print(f"不确定性倍数: {scenario_result.matched_scenario.uncertainty_multiplier}x")
            print(f"场景描述: {scenario_result.matched_scenario.description}")
            print(f"电力系统影响: {scenario_result.matched_scenario.power_system_impact}")
            print(f"运行建议: {scenario_result.matched_scenario.operation_suggestions}")
            print("="*50)
            
            return {"success": True, "scenario_result": scenario_result}
            
        except Exception as e:
            error_msg = f"❌ 场景分析失败: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def test_system_health(self):
        """系统健康检查"""
        print("🔍 执行系统健康检查...")
        
        tests = [
            ("DDD容器", lambda: self.container is not None),
            ("预测服务", lambda: self.container.forecast_service is not None),
            ("不确定性服务", lambda: self.container.uncertainty_service is not None),
            ("预测仓储", lambda: self.container.forecast_repo is not None),
            ("模型仓储", lambda: self.container.model_repo is not None),
            ("场景仓储", lambda: self.container.weather_scenario_repo is not None),
        ]
        
        print("\n健康检查结果:")
        all_healthy = True
        for name, test in tests:
            try:
                result = test()
                status = "✅ 正常" if result else "❌ 异常"
                print(f"  {name}: {status}")
                if not result:
                    all_healthy = False
            except Exception as e:
                print(f"  {name}: ❌ 错误 - {e}")
                all_healthy = False
        
        if all_healthy:
            print("\n🎉 系统状态良好，可以正常使用！")
        else:
            print("\n⚠️ 系统存在问题，请检查配置")
        
        return all_healthy

    def run_training_demo(self, province: str = "上海"):
        """运行DDD训练功能演示"""
        print(f"🏭 开始为 {province} 进行DDD训练演示...")
        
        try:
            # 导入训练相关组件
            from AveMujica_DDD.application.services.training_service import TrainingService
            from AveMujica_DDD.application.dtos.training_dto import TrainingRequestDTO
            from AveMujica_DDD.infrastructure.repositories.training_task_repository import FileTrainingTaskRepository
            from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
            
            # 创建简单的模拟模型仓储
            class MockModelRepository(IModelRepository):
                def save(self, model): pass
                def find_by_id(self, model_id): return None
                def delete(self, model_id): pass
                def find_by_type_and_region(self, model_type, region): return []
            
            # 创建训练服务
            task_repo = FileTrainingTaskRepository("demo_training")
            model_repo = MockModelRepository()
            training_service = TrainingService(task_repo, model_repo)
            
            print("✅ DDD训练服务创建成功")
            
            # 创建训练请求
            request = TrainingRequestDTO(
                model_type="convtrans",
                forecast_type="load",
                province=province,
                train_start_date="2024-01-01",
                train_end_date="2024-01-15",  # 缩短训练期间用于演示
                epochs=3,  # 减少轮数用于演示
                batch_size=16,
                learning_rate=0.001
            )
            
            print(f"✅ 训练请求已创建: {request.model_type} 模型 - {province}")
            
            # 创建训练任务
            task_id = training_service.create_training_task(request)
            print(f"✅ 训练任务创建成功: {task_id}")
            
            # 执行训练
            print("\n🔧 开始执行DDD训练...")
            print("注意: 这是演示模式，使用模拟数据")
            
            training_result = training_service.execute_training(task_id)
            
            print("\n" + "="*50)
            print("🎯 DDD训练结果")
            print("="*50)
            print(f"任务ID: {training_result.task_id}")
            print(f"训练状态: {training_result.status}")
            print(f"模型类型: {training_result.model_type}")
            print(f"预测类型: {training_result.forecast_type}")
            print(f"省份: {training_result.province}")
            print(f"模型路径: {training_result.model_path}")
            
            # 显示训练指标
            print(f"\n📊 训练指标:")
            metrics = {
                "MAE": training_result.mae,
                "RMSE": training_result.rmse,
                "MAPE": training_result.mape,
                "最终损失": training_result.final_loss,
                "最佳验证损失": training_result.best_val_loss
            }
            for key, value in metrics.items():
                if value is not None:
                    print(f"   {key}: {value}")
            
            # 查询训练历史
            print(f"\n📚 训练历史:")
            history = training_service.list_training_history(limit=3)
            print(f"   总任务数: {history.total_tasks}")
            print(f"   完成任务数: {history.completed_tasks}")
            print(f"   成功率: {history.get_success_rate():.1f}%")
            
            print("="*50)
            
            # 清理演示文件
            import shutil
            if os.path.exists("demo_training"):
                shutil.rmtree("demo_training")
            
            return {"success": True, "training_result": training_result}
            
        except Exception as e:
            error_msg = f"❌ DDD训练演示失败: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}


def main():
    """主函数 - 优雅的命令行界面"""
    parser = argparse.ArgumentParser(
        description="🌟 MUJICA DDD 天气感知负荷预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python start_system.py --demo               # 运行预测演示
  python start_system.py --api               # 启动API服务器
  python start_system.py --scenario          # 场景分析演示
  python start_system.py --training          # DDD训练功能演示
  python start_system.py --health            # 系统健康检查
  python start_system.py --full              # 完整演示
        """
    )
    
    parser.add_argument("--demo", action="store_true", help="运行预测演示")
    parser.add_argument("--api", action="store_true", help="启动API服务器")
    parser.add_argument("--scenario", action="store_true", help="场景分析演示")
    parser.add_argument("--training", action="store_true", help="DDD训练功能演示")
    parser.add_argument("--health", action="store_true", help="系统健康检查")
    parser.add_argument("--full", action="store_true", help="完整演示")
    parser.add_argument("--province", default="上海", help="省份名称 (默认: 上海)")
    parser.add_argument("--port", type=int, default=5001, help="API端口 (默认: 5001)")
    parser.add_argument("--memory", action="store_true", help="使用内存实现（测试模式）")
    
    args = parser.parse_args()
    
    # 如果没有参数，显示帮助
    if not any([args.demo, args.api, args.scenario, args.training, args.health, args.full]):
        parser.print_help()
        return
    
    print("🌟 MUJICA 天气感知负荷预测系统 - DDD架构版本")
    print("="*60)
    
    # 初始化系统
    launcher = MujicaDDDLauncher(use_real_implementations=not args.memory)
    
    try:
        if args.health or args.full:
            launcher.test_system_health()
            print()
        
        if args.scenario or args.full:
            launcher.run_scenario_analysis_demo(args.province)
            print()
        
        if args.training or args.full:
            launcher.run_training_demo(args.province)
            print()
        
        if args.demo or args.full:
            launcher.run_forecast_demo(args.province)
            print()
        
        if args.api:
            print("🌐 API服务器启动说明:")
            print(f"   访问地址: http://localhost:{args.port}")
            print("   健康检查: http://localhost:{}/api/health".format(args.port))
            print("   API文档将在控制台显示\n")
            launcher.start_api_server(port=args.port)
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，系统已退出")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 