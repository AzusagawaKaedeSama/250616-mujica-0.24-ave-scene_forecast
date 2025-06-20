#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🌟 MUJICA 交互式系统
简化的交互界面，用于命令下发和功能测试
"""

import sys
import os
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def show_menu():
    """显示交互菜单"""
    print("\n🌟 MUJICA 系统启动器")
    print("=" * 50)
    print("请选择操作:")
    print("1. 启动原系统 (app.py + 前端)")
    print("2. 启动DDD API服务器")
    print("3. 启动前端服务器") 
    print("4. 同时启动DDD API+前端")
    print("5. 预测功能演示")
    print("6. 训练功能演示") 
    print("7. 系统健康检查")
    print("8. 完整功能演示")
    print("9. 退出")
    print("=" * 50)

def start_api_server():
    """启动API服务器"""
    print("🚀 启动API服务器...")
    
    try:
        from AveMujica_DDD.api import DIContainer, create_app
        
        print("🔧 初始化系统...")
        container = DIContainer(use_real_implementations=False)
        print("✅ 系统初始化完成")
        
        print("🔧 创建Flask应用...")
        app = create_app(container)
        print("✅ Flask应用创建成功")
        
        print("\n🌐 API服务器信息:")
        print("   地址: http://localhost:5001")
        print("   健康检查: http://localhost:5001/api/health")
        print("   按 Ctrl+C 停止服务器")
        print()
        
        app.run(host="0.0.0.0", port=5001, debug=True)
        
    except Exception as e:
        print(f"❌ API服务器启动失败: {e}")
        import traceback
        traceback.print_exc()

def start_original_system():
    """启动原系统 (app.py + 前端)"""
    print("🔥 启动原MUJICA系统...")
    
    try:
        import subprocess
        import threading
        import time
        import os
        
        print("🔧 启动原系统后端 (app.py)...")
        
        # 启动app.py（后台）
        def start_app_background():
            try:
                subprocess.run(["python", "app.py"], check=True)
            except Exception as e:
                print(f"app.py启动错误: {e}")
        
        app_thread = threading.Thread(target=start_app_background)
        app_thread.daemon = True
        app_thread.start()
        
        print("✅ 原系统后端已启动")
        time.sleep(3)  # 等待后端启动
        
        print("🔧 启动前端开发服务器...")
        
        # 启动前端服务器
        frontend_dir = os.path.join(os.getcwd(), "frontend")
        
        print("\n🌐 原系统信息:")
        print("   后端API: http://localhost:5001")
        print("   前端应用: http://localhost:5173")
        print("   这是您之前使用的系统配置")
        print("   按 Ctrl+C 停止所有服务器")
        print()
        
        # 检查Node.js环境
        try:
            result = subprocess.run(["npm", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Node.js环境正常 (npm {result.stdout.strip()})")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ 未找到npm命令")
            print("💡 请先安装Node.js，然后在frontend目录运行: npm install")
            return
        
        # 检查依赖
        node_modules = os.path.join(frontend_dir, "node_modules")
        if not os.path.exists(node_modules):
            print("⚠️ 前端依赖未安装，正在安装...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            print("✅ 前端依赖安装完成")
        
        # 启动前端开发服务器
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir)
            
    except Exception as e:
        print(f"❌ 原系统启动失败: {e}")
        print("\n💡 手动启动方式:")
        print("   1. 终端1: python app.py")
        print("   2. 终端2: cd frontend && npm run dev")
        print("   3. 访问: http://localhost:5173")

def start_frontend_server():
    """启动前端服务器"""
    print("🌐 启动前端服务器...")
    
    try:
        import subprocess
        import os
        
        frontend_dir = os.path.join(os.getcwd(), "frontend")
        frontend_script = os.path.join(frontend_dir, "start_frontend.py")
        
        print(f"📁 前端目录: {frontend_dir}")
        print("🎯 智能选择最佳前端技术栈...")
        
        if os.path.exists(frontend_script):
            print("✅ 使用智能前端启动器")
            print("\n📋 启动选项说明:")
            print("   1️⃣ Vite开发服务器 - 完整React体验 (http://localhost:5173)")
            print("   2️⃣ 静态文件服务器 - 仪表板体验 (http://localhost:8080)")
            print("   3️⃣ 自动选择最佳方案")
            print()
            
            choice = input("请选择启动方式 (1/2/3，默认3): ").strip()
            
            if choice == "1":
                # 仅尝试Vite开发服务器
                subprocess.run(["python", frontend_script, "--vite-only"], cwd=frontend_dir)
            elif choice == "2":
                # 强制使用简单HTTP服务器
                subprocess.run(["python", frontend_script, "--simple"], cwd=frontend_dir)
            else:
                # 智能选择（默认）
                subprocess.run(["python", frontend_script], cwd=frontend_dir)
        else:
            # 备用方案：直接使用Python HTTP服务器
            print("⚠️ 未找到前端启动脚本，使用备用方案...")
            print("🔧 启动简单HTTP服务器...")
            print("\n🌐 前端服务器信息:")
            print("   地址: http://localhost:8080")
            print("   推荐访问: http://localhost:8080/dashboard.html")
            print("   按 Ctrl+C 停止服务器")
            print()
            
            os.chdir(frontend_dir)
            subprocess.run(["python", "-m", "http.server", "8080"])
            
    except Exception as e:
        print(f"❌ 前端服务器启动失败: {e}")
        print("\n💡 解决方案:")
        print("   1. 确保已安装Node.js和npm")
        print("   2. 在frontend目录运行: npm install")
        print("   3. 或直接访问: frontend/dashboard.html")
        import traceback
        traceback.print_exc()

def start_both_servers():
    """同时启动API和前端服务器"""
    print("🚀 同时启动API和前端服务器...")
    
    try:
        import subprocess
        import threading
        import time
        import os
        
        print("🔧 启动API服务器（后台）...")
        
        # 启动API服务器（后台）
        def start_api_background():
            try:
                from AveMujica_DDD.api import DIContainer, create_app
                container = DIContainer(use_real_implementations=False)
                app = create_app(container)
                app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
            except Exception as e:
                print(f"API服务器错误: {e}")
        
        api_thread = threading.Thread(target=start_api_background)
        api_thread.daemon = True
        api_thread.start()
        
        print("✅ API服务器已在后台启动")
        time.sleep(2)  # 等待API服务器启动
        
        print("🔧 启动前端服务器...")
        
        # 启动前端服务器（前台）
        frontend_dir = os.path.join(os.getcwd(), "frontend")
        frontend_script = os.path.join(frontend_dir, "start_frontend.py")
        
        print("\n🌐 服务器信息:")
        print("   API服务器: http://localhost:5001")
        print("   前端服务器: http://localhost:8080")
        print("   预测仪表板: http://localhost:8080/dashboard.html")
        print("   按 Ctrl+C 停止所有服务器")
        print()
        
        if os.path.exists(frontend_script):
            subprocess.run(["python", frontend_script, "--no-browser"], cwd=frontend_dir)
        else:
            # 备用方案：使用Python内置HTTP服务器
            os.chdir(frontend_dir)
            subprocess.run(["python", "-m", "http.server", "8080"])
            
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

def run_forecast_demo():
    """运行预测功能演示"""
    print("📊 运行预测功能演示...")
    
    try:
        from AveMujica_DDD.api import DIContainer
        from AveMujica_DDD.domain.aggregates.prediction_model import ForecastType
        from datetime import date, timedelta
        
        print("🔧 初始化系统...")
        container = DIContainer(use_real_implementations=False)
        
        start_date = date.today() + timedelta(days=1)
        end_date = start_date
        province = "上海"
        
        print(f"🔮 开始预测: {province} - {start_date}")
        
        forecast_dto = container.forecast_service.create_day_ahead_forecast(
            province=province,
            start_date=start_date,
            end_date=end_date,
            forecast_type=ForecastType.LOAD
        )
        
        print("\n✅ 预测完成!")
        print(f"   预测ID: {forecast_dto.forecast_id}")
        print(f"   省份: {forecast_dto.province}")
        print(f"   模型: {forecast_dto.model_name}")
        print(f"   场景: {forecast_dto.scenario_type}")
        print(f"   数据点数: {len(forecast_dto.time_series)}")
        
        if forecast_dto.time_series:
            avg_value = sum(p.value for p in forecast_dto.time_series) / len(forecast_dto.time_series)
            print(f"   平均预测值: {avg_value:.1f} MW")
        
    except Exception as e:
        print(f"❌ 预测演示失败: {e}")
        import traceback
        traceback.print_exc()

def run_training_demo():
    """运行训练功能演示"""
    print("🏭 运行训练功能演示...")
    
    try:
        from AveMujica_DDD.tests.integration.test_training_end_to_end import test_end_to_end_training
        success = test_end_to_end_training()
        
        if success:
            print("✅ 训练演示成功完成!")
        else:
            print("⚠️ 训练演示部分失败")
            
    except Exception as e:
        print(f"❌ 训练演示失败: {e}")
        import traceback
        traceback.print_exc()

def run_health_check():
    """运行系统健康检查"""
    print("🏥 运行系统健康检查...")
    
    try:
        from AveMujica_DDD.api import DIContainer
        
        print("🔧 检查系统组件...")
        container = DIContainer(use_real_implementations=False)
        
        checks = [
            ("DDD容器", lambda: container is not None),
            ("预测服务", lambda: container.forecast_service is not None),
            ("不确定性服务", lambda: container.uncertainty_service is not None),
            ("天气场景仓储", lambda: container.weather_scenario_repo is not None),
            ("模型仓储", lambda: container.model_repo is not None),
        ]
        
        print("\n📋 健康检查结果:")
        all_healthy = True
        for name, check in checks:
            try:
                result = check()
                status = "✅ 正常" if result else "❌ 异常"
                print(f"   {name}: {status}")
                if not result:
                    all_healthy = False
            except Exception as e:
                print(f"   {name}: ❌ 错误 - {e}")
                all_healthy = False
        
        if all_healthy:
            print("\n🎉 系统状态良好!")
        else:
            print("\n⚠️ 系统存在问题")
            
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        import traceback
        traceback.print_exc()

def run_full_demo():
    """运行完整功能演示"""
    print("🎪 运行完整功能演示...")
    
    demos = [
        ("系统健康检查", run_health_check),
        ("预测功能演示", run_forecast_demo),
        ("训练功能演示", run_training_demo)
    ]
    
    passed = 0
    total = len(demos)
    
    for name, demo_func in demos:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            demo_func()
            passed += 1
            print(f"✅ {name} 完成")
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
    
    print(f"\n🎯 演示总结: {passed}/{total} 个功能正常")

def main():
    """主函数"""
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 工作目录: {os.getcwd()}")
    
    while True:
        try:
            show_menu()
            choice = input("\n请输入选择 (1-9): ").strip()
            
            if choice == "1":
                start_original_system()
            elif choice == "2":
                start_api_server()
            elif choice == "3":
                start_frontend_server()
            elif choice == "4":
                start_both_servers()
            elif choice == "5":
                run_forecast_demo()
            elif choice == "6":
                run_training_demo()
            elif choice == "7":
                run_health_check()
            elif choice == "8":
                run_full_demo()
            elif choice == "9":
                print("\n👋 再见！")
                break
            else:
                print("\n❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，程序退出")
            break
        except Exception as e:
            print(f"\n💥 程序异常: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 