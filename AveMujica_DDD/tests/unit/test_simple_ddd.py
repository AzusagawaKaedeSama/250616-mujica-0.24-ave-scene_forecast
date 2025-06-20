#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化的DDD基本功能测试
只测试核心导入和基本功能
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_basic_imports():
    """测试基本导入是否正常"""
    print("📦 测试基本导入...")
    
    try:
        # 测试API容器导入
        from AveMujica_DDD.api import DIContainer
        print("✅ API容器导入成功")
        
        # 测试基本服务导入
        from AveMujica_DDD.application.services.forecast_service import ForecastService
        print("✅ 预测服务导入成功")
        
        # 测试内存仓储导入
        from AveMujica_DDD.infrastructure.repositories.in_memory_repos import InMemoryForecastRepository
        print("✅ 内存仓储导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 基本导入测试失败: {e}")
        return False

def test_container_creation():
    """测试容器创建"""
    print("🔗 测试容器创建...")
    
    try:
        from AveMujica_DDD.api import DIContainer
        
        # 创建内存容器
        container = DIContainer(use_real_implementations=False)
        assert container is not None
        assert container.forecast_service is not None
        print("✅ 内存容器创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 容器创建测试失败: {e}")
        return False

def test_repository_basic():
    """测试仓储基本功能"""
    print("🗄️ 测试仓储基本功能...")
    
    try:
        from AveMujica_DDD.infrastructure.repositories.in_memory_repos import InMemoryForecastRepository
        
        repo = InMemoryForecastRepository()
        assert repo is not None
        print("✅ 仓储创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 仓储测试失败: {e}")
        return False

def test_training_service_import():
    """测试训练服务导入"""
    print("🏭 测试训练服务导入...")
    
    try:
        from AveMujica_DDD.application.services.training_service import TrainingService
        from AveMujica_DDD.application.dtos.training_dto import TrainingRequestDTO
        print("✅ 训练服务导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 训练服务导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 简化DDD基本功能测试")
    print("=" * 40)
    
    tests = [
        ("基本导入", test_basic_imports),
        ("容器创建", test_container_creation),
        ("仓储功能", test_repository_basic),
        ("训练服务", test_training_service_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📋 测试 {name}...")
        if test_func():
            passed += 1
            print(f"✅ {name} 通过")
        else:
            print(f"❌ {name} 失败")
    
    print("\n" + "=" * 40)
    print(f"🎯 结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        return True
    else:
        print("⚠️ 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    print(f"退出码: {0 if success else 1}")
    sys.exit(0 if success else 1) 