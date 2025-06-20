#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🌟 MUJICA DDD 系统启动器
优雅启动重构后的四层DDD架构系统
集成预测、训练、场景分析等所有功能
"""

import sys
import os
import argparse
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class MujicaDDDLauncher:
    """MUJICA DDD系统启动器"""
    
    def __init__(self):
        print("🌟 初始化 MUJICA DDD 系统...")
        
    def run_api_server(self, port: int = 5001):
        """启动API服务器"""
        print(f"🚀 启动API服务器: http://localhost:{port}")
        
        try:
            from AveMujica_DDD.api import DIContainer, create_app
            
            # 创建容器（使用内存实现，避免导入错误）
            container = DIContainer(use_real_implementations=False)
            app = create_app(container)
            
            print("✅ API服务器准备就绪")
            print(f"📝 访问地址: http://localhost:{port}/api/health")
            
            app.run(host="0.0.0.0", port=port, debug=True)
            
        except Exception as e:
            print(f"❌ API服务器启动失败: {e}")
            return False
    
    def run_training_demo(self, province: str = "上海"):
        """运行训练演示"""
        print(f"🏭 启动 {province} 的训练演示...")
        
        try:
            from AveMujica_DDD.tests.integration.test_training_end_to_end import test_end_to_end_training
            return test_end_to_end_training()
        except Exception as e:
            print(f"❌ 训练演示失败: {e}")
            return False
    
    def run_forecast_demo(self, province: str = "上海"):
        """运行预测演示"""
        print(f"📊 启动 {province} 的预测演示...")
        
        try:
            from AveMujica_DDD.api import DIContainer
            from AveMujica_DDD.domain.aggregates.prediction_model import ForecastType
            from datetime import date, timedelta
            
            container = DIContainer(use_real_implementations=False)
            
            start_date = date.today() + timedelta(days=1)
            end_date = start_date
            
            print(f"预测日期: {start_date}")
            
            # 调用预测服务
            forecast_dto = container.forecast_service.create_day_ahead_forecast(
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
            print(f"模型: {forecast_dto.model_name}")
            print(f"场景: {forecast_dto.scenario_type}")
            print(f"数据点数: {len(forecast_dto.time_series)}")
            
            if forecast_dto.time_series:
                avg_value = sum(p.value for p in forecast_dto.time_series) / len(forecast_dto.time_series)
                print(f"平均预测值: {avg_value:.1f} MW")
            
            print("="*50)
            return True
            
        except Exception as e:
            print(f"❌ 预测演示失败: {e}")
            return False
    
    def run_system_health_check(self):
        """系统健康检查"""
        print("🏥 执行系统健康检查...")
        
        try:
            from AveMujica_DDD.tests.test_runner import run_system_health_check
            return run_system_health_check()
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            return False
    
    def run_full_demo(self, province: str = "上海"):
        """运行完整演示"""
        print(f"🎪 运行 {province} 的完整系统演示...")
        
        demos = [
            ("系统健康检查", lambda: self.run_system_health_check()),
            ("预测功能演示", lambda: self.run_forecast_demo(province)),
            ("训练功能演示", lambda: self.run_training_demo(province))
        ]
        
        passed = 0
        total = len(demos)
        
        for name, demo_func in demos:
            print(f"\n📋 执行 {name}...")
            if demo_func():
                passed += 1
                print(f"✅ {name} 成功")
            else:
                print(f"❌ {name} 失败")
        
        print(f"\n🎯 演示总结: {passed}/{total} 个功能正常")
        return passed == total

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="🌟 MUJICA DDD 天气感知负荷预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python launcher.py --api                    # 启动API服务器
  python launcher.py --forecast               # 预测演示
  python launcher.py --training               # 训练演示
  python launcher.py --health                 # 健康检查
  python launcher.py --demo                   # 完整演示
        """
    )
    
    parser.add_argument("--api", action="store_true", help="启动API服务器")
    parser.add_argument("--forecast", action="store_true", help="预测演示")
    parser.add_argument("--training", action="store_true", help="训练演示")
    parser.add_argument("--health", action="store_true", help="系统健康检查")
    parser.add_argument("--demo", action="store_true", help="完整演示")
    parser.add_argument("--province", default="上海", help="省份名称")
    parser.add_argument("--port", type=int, default=5001, help="API端口")
    
    args = parser.parse_args()
    
    # 如果没有参数，显示交互式菜单
    if not any([args.api, args.forecast, args.training, args.health, args.demo]):
        print("🌟 MUJICA DDD 天气感知负荷预测系统")
        print("=" * 50)
        print("请选择操作:")
        print("1. 启动API服务器")
        print("2. 预测功能演示")
        print("3. 训练功能演示")
        print("4. 系统健康检查")
        print("5. 完整功能演示")
        print("6. 退出")
        
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == "1":
            args.api = True
        elif choice == "2":
            args.forecast = True
        elif choice == "3":
            args.training = True
        elif choice == "4":
            args.health = True
        elif choice == "5":
            args.demo = True
        elif choice == "6":
            print("👋 再见！")
            return
        else:
            print("❌ 无效选择")
            return
    
    print("🌟 MUJICA DDD 系统启动")
    print("=" * 50)
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    launcher = MujicaDDDLauncher()
    
    try:
        if args.health:
            launcher.run_system_health_check()
        
        if args.forecast:
            launcher.run_forecast_demo(args.province)
        
        if args.training:
            launcher.run_training_demo(args.province)
        
        if args.demo:
            launcher.run_full_demo(args.province)
        
        if args.api:
            print("🌐 启动API服务器...")
            print(f"   访问地址: http://localhost:{args.port}")
            print("   按 Ctrl+C 停止服务器\n")
            launcher.run_api_server(args.port)
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，系统已退出")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 