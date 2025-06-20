#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优雅的DDD测试运行器
统一管理所有测试，保持根目录整洁
"""

import sys
import os
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_unit_tests():
    """运行单元测试"""
    print("🔧 运行单元测试...")
    
    try:
        from AveMujica_DDD.tests.unit.test_simple_ddd import main as run_simple_test
        return run_simple_test()
    except Exception as e:
        print(f"❌ 单元测试失败: {e}")
        return False

def run_integration_tests():
    """运行集成测试"""
    print("🔗 运行集成测试...")
    
    try:
        from AveMujica_DDD.tests.integration.test_training_end_to_end import main as run_e2e_test
        return run_e2e_test()
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def run_system_health_check():
    """运行系统健康检查"""
    print("🏥 系统健康检查...")
    
    try:
        from AveMujica_DDD.api import DIContainer
        
        # 测试内存实现
        container = DIContainer(use_real_implementations=False)
        
        checks = [
            ("依赖注入容器", lambda: container is not None),
            ("预测服务", lambda: container.forecast_service is not None),
            ("不确定性服务", lambda: container.uncertainty_service is not None),
        ]
        
        all_healthy = True
        for name, check in checks:
            try:
                result = check()
                status = "✅" if result else "❌"
                print(f"  {name}: {status}")
                if not result:
                    all_healthy = False
            except Exception as e:
                print(f"  {name}: ❌ 错误 - {e}")
                all_healthy = False
        
        return all_healthy
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🌟 MUJICA DDD 测试套件")
    print("=" * 50)
    print(f"⏰ 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 工作目录: {os.getcwd()}")
    print()
    
    # 询问用户想要运行的测试
    print("请选择要运行的测试:")
    print("1. 系统健康检查 (快速)")
    print("2. 单元测试")
    print("3. 集成测试 (包含训练功能)")
    print("4. 全部测试")
    print("5. 退出")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == "1":
        print("\n🏥 执行系统健康检查...")
        success = run_system_health_check()
        
    elif choice == "2":
        print("\n🔧 执行单元测试...")
        success = run_unit_tests()
        
    elif choice == "3":
        print("\n🔗 执行集成测试...")
        success = run_integration_tests()
        
    elif choice == "4":
        print("\n🚀 执行全部测试...")
        tests = [
            ("系统健康检查", run_system_health_check),
            ("单元测试", run_unit_tests),
            ("集成测试", run_integration_tests)
        ]
        
        passed = 0
        total = len(tests)
        
        for name, test_func in tests:
            print(f"\n📋 运行 {name}...")
            if test_func():
                passed += 1
                print(f"✅ {name} 通过")
            else:
                print(f"❌ {name} 失败")
        
        success = (passed == total)
        print(f"\n🎯 总结: {passed}/{total} 个测试套件通过")
        
    elif choice == "5":
        print("\n👋 退出测试")
        return True
    else:
        print("\n❌ 无效选择")
        return False
    
    # 总结
    print("\n" + "=" * 50)
    if success:
        print("🎉 测试完成，系统状态良好！")
        print("✅ DDD架构运行正常，可以投入使用")
    else:
        print("⚠️ 测试发现问题，请检查系统配置")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试运行器异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 