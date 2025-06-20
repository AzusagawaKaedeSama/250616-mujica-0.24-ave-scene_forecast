#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DDD核心功能单元测试
测试领域层、应用层、基础设施层的基本功能
"""

import sys
import os
from datetime import date, datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_domain_layer():
    """测试领域层基本功能"""
    print("🏗️ 测试领域层...")
    
    try:
        # 测试天气场景
        from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario
        
        # 创建基本场景
        from AveMujica_DDD.domain.aggregates.weather_scenario import ScenarioType
        scenario = WeatherScenario(
            scenario_type=ScenarioType.MODERATE_NORMAL,
            description="正常天气",
            uncertainty_multiplier=1.0,
            typical_features={"temperature": 20.0, "humidity": 60.0, "wind_speed": 4.0, "precipitation": 0.0},
            power_system_impact="系统平稳运行",
            operation_suggestions="标准运行模式"
        )
        assert scenario.uncertainty_multiplier == 1.0
        print("✅ 天气场景测试通过")
        
        # 测试预测模型
        from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
        import uuid
        
        model = PredictionModel(
            model_id=uuid.uuid4(),
            name="test_model",
            forecast_type=ForecastType.LOAD,
            file_path="test_path"
        )
        assert model.name == "test_model"
        assert model.forecast_type == ForecastType.LOAD
        print("✅ 预测模型测试通过")
        
        return True
    except Exception as e:
        print(f"❌ 领域层测试失败: {e}")
        return False

def test_application_layer():
    """测试应用层基本功能"""
    print("🎯 测试应用层...")
    
    try:
        # 测试DTO
        from AveMujica_DDD.application.dtos.forecast_dto import ForecastRequestDTO
        
        request = ForecastRequestDTO(
            province="上海",
            start_date=date.today(),
            end_date=date.today(),
            forecast_type="load"
        )
        assert request.province == "上海"
        assert request.forecast_type == "load"
        print("✅ DTO测试通过")
        
        return True
    except Exception as e:
        print(f"❌ 应用层测试失败: {e}")
        return False

def test_infrastructure_layer():
    """测试基础设施层基本功能"""
    print("🔧 测试基础设施层...")
    
    try:
        # 测试内存仓储
        from AveMujica_DDD.infrastructure.repositories.in_memory_repos import InMemoryForecastRepository
        
        repo = InMemoryForecastRepository()
        assert repo is not None
        print("✅ 内存仓储测试通过")
        
        return True
    except Exception as e:
        print(f"❌ 基础设施层测试失败: {e}")
        return False

def test_dependency_injection():
    """测试依赖注入容器"""
    print("🔗 测试依赖注入...")
    
    try:
        from AveMujica_DDD.api import DIContainer
        
        # 测试内存实现
        container = DIContainer(use_real_implementations=False)
        assert container.forecast_service is not None
        assert container.uncertainty_service is not None
        print("✅ 依赖注入测试通过")
        
        return True
    except Exception as e:
        print(f"❌ 依赖注入测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始DDD核心功能单元测试")
    print("=" * 50)
    
    tests = [
        ("领域层", test_domain_layer),
        ("应用层", test_application_layer),
        ("基础设施层", test_infrastructure_layer),
        ("依赖注入", test_dependency_injection)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📋 测试 {name}...")
        if test_func():
            passed += 1
            print(f"✅ {name} 测试通过")
        else:
            print(f"❌ {name} 测试失败")
    
    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有单元测试都通过了！")
        return True
    else:
        print("⚠️ 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 