#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUJICA DDD系统启动脚本
用于启动完整的多源数据负荷预测系统
"""

import os
import sys
import threading
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from AveMujica_DDD.main import MujicaDDDSystem
from AveMujica_DDD.api import DIContainer, create_app


def print_banner():
    """打印系统启动横幅"""
    print("=" * 80)
    print("🌟 MUJICA 多源数据负荷预测系统 - DDD架构版本")
    print("   天气感知 | 区间预测 | 不确定性量化")
    print("=" * 80)


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    # 检查关键依赖
    try:
        import torch
        import pandas
        import numpy
        import flask
        print("✅ 关键依赖已安装")
        
        # 检查PyTorch设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ PyTorch设备: {device}")
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return False
    
    return True


def check_models_directory():
    """检查模型目录结构"""
    print("📁 检查模型目录...")
    
    models_dir = project_root / "models"
    if not models_dir.exists():
        print("❌ models目录不存在")
        return False
    
    # 检查convtrans_weather目录
    convtrans_dir = models_dir / "convtrans_weather"
    if convtrans_dir.exists():
        model_count = 0
        for forecast_type in ['load', 'pv', 'wind']:
            type_dir = convtrans_dir / forecast_type
            if type_dir.exists():
                for province_dir in type_dir.iterdir():
                    if province_dir.is_dir():
                        # 检查模型文件
                        if (province_dir / "best_model.pth").exists() or \
                           (province_dir / "convtrans_weather_model.pth").exists():
                            model_count += 1
        
        print(f"✅ 发现 {model_count} 个训练好的模型")
        return model_count > 0
    else:
        print("⚠️  没有发现现有的训练模型，将使用合成预测")
        return True


def start_api_server(port=5001):
    """启动API服务器"""
    print(f"🚀 启动API服务器 (端口: {port})...")
    
    try:
        # 创建DI容器
        di_container = DIContainer(use_real_implementations=True)
        
        # 创建Flask应用
        app = create_app(di_container)
        
        print(f"✅ API服务器启动成功: http://localhost:{port}")
        print("   可用端点:")
        print("   - GET  /api/health      - 健康检查")
        print("   - POST /api/predict     - 执行预测") 
        print("   - POST /api/train       - 训练模型")
        print("   - GET  /api/models      - 模型列表")
        print("   - GET  /api/provinces   - 支持省份")
        print("   - GET  /api/scenarios   - 天气场景")
        
        # 启动服务器
        app.run(debug=False, port=port, host='0.0.0.0', use_reloader=False)
        
    except Exception as e:
        print(f"❌ API服务器启动失败: {e}")
        import traceback
        traceback.print_exc()


def run_demo_forecast(system: MujicaDDDSystem):
    """运行演示预测"""
    print("\n📊 运行演示预测...")
    
    try:
        # 运行上海的负荷预测演示
        result = system.run_example_forecast(province="上海", days_ahead=1)
        
        if result:
            print("✅ 演示预测完成")
            return True
        else:
            print("❌ 演示预测失败")
            return False
            
    except Exception as e:
        print(f"❌ 演示预测错误: {e}")
        return False


def main():
    """主函数"""
    print_banner()
    
    # 1. 环境检查
    if not check_environment():
        print("❌ 环境检查失败，退出系统")
        return 1
    
    # 2. 模型目录检查
    if not check_models_directory():
        print("❌ 模型目录检查失败，退出系统")
        return 1
    
    # 3. 初始化系统
    print("🔧 初始化DDD系统...")
    try:
        system = MujicaDDDSystem(use_real_implementations=True)
        print("✅ DDD系统初始化成功")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 4. 显示系统状态
    print("\n📋 系统状态:")
    system.list_available_models()
    system.list_weather_scenarios()
    
    # 5. 选择启动模式
    print("\n" + "=" * 60)
    print("请选择启动模式:")
    print("1. 完整演示 (先运行预测演示，然后启动API)")
    print("2. 仅启动API服务器")
    print("3. 仅运行预测演示")
    print("4. 退出")
    
    try:
        choice = input("请输入选择 (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n👋 用户取消，退出系统")
        return 0
    
    if choice == '1':
        # 完整演示模式
        if run_demo_forecast(system):
            print("\n" + "=" * 60)
            input("按Enter键启动API服务器...")
            start_api_server()
        else:
            print("演示失败，是否仍要启动API服务器？(y/N): ", end="")
            if input().lower().startswith('y'):
                start_api_server()
    
    elif choice == '2':
        # 仅API服务器
        start_api_server()
    
    elif choice == '3':
        # 仅预测演示
        run_demo_forecast(system)
        
    elif choice == '4':
        print("👋 退出系统")
        return 0
    
    else:
        print("❌ 无效选择，退出系统")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 