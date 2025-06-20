#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUJICA DDD系统 - 快速启动API服务器
用于直接启动API服务，供前端Web界面使用
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from AveMujica_DDD.api import DIContainer, create_app


def main():
    """快速启动API服务器"""
    print("🚀 启动MUJICA DDD API服务器...")
    print("   端口: 5001")
    print("   CORS: 启用")
    print("   实现: 真实预测引擎 + 文件系统仓储")
    print("=" * 50)
    
    try:
        # 创建DI容器（使用真实实现）
        print("初始化依赖注入容器...")
        di_container = DIContainer(use_real_implementations=True)
        
        # 创建Flask应用
        print("创建Flask应用...")
        app = create_app(di_container)
        
        print("✅ 服务器准备就绪!")
        print("\n📍 API端点:")
        print("   - GET  /api/health      - 健康检查")
        print("   - POST /api/predict     - 执行预测")
        print("   - POST /api/train       - 训练模型") 
        print("   - GET  /api/models      - 模型列表")
        print("   - GET  /api/provinces   - 支持省份")
        print("   - GET  /api/scenarios   - 天气场景")
        print("   - GET  /api/historical-results - 历史结果")
        
        print(f"\n🌐 访问地址: http://localhost:5001")
        print("   前端应该能够连接到此API服务器")
        print("\n按 Ctrl+C 停止服务器")
        print("=" * 50)
        
        # 启动服务器
        app.run(
            debug=False,
            port=5001,
            host='0.0.0.0',
            use_reloader=False  # 避免重载器问题
        )
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 