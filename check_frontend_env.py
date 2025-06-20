#!/usr/bin/env python3
"""
🔍 前端环境检查工具
检查Node.js、npm和前端依赖是否正确安装
"""

import subprocess
import os
import sys
from pathlib import Path

def check_command(command, description):
    """检查命令是否可用"""
    try:
        result = subprocess.run([command, "--version"], 
                              capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print(f"✅ {description}: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"❌ {description}: 未安装或无法访问")
        return False

def check_frontend_setup():
    """检查前端设置"""
    print("🔍 MUJICA 前端环境检查")
    print("="*40)
    
    # 检查基础工具
    print("\n📋 基础工具检查:")
    node_ok = check_command("node", "Node.js")
    npm_ok = check_command("npm", "npm包管理器")
    
    if not (node_ok and npm_ok):
        print("\n💡 安装建议:")
        print("   1. 访问 https://nodejs.org 下载Node.js")
        print("   2. 安装后重启终端")
        print("   3. 运行 node --version 和 npm --version 验证")
        return False
    
    # 检查前端目录
    print("\n📁 前端项目检查:")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("❌ frontend目录不存在")
        return False
    
    print("✅ frontend目录存在")
    
    # 检查package.json
    package_json = frontend_dir / "package.json"
    if package_json.exists():
        print("✅ package.json存在")
    else:
        print("❌ package.json不存在")
        return False
    
    # 检查node_modules
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        print("✅ node_modules已安装")
        dependencies_ok = True
    else:
        print("⚠️ node_modules未安装")
        dependencies_ok = False
    
    # 检查dist目录
    dist_dir = frontend_dir / "dist"
    if dist_dir.exists():
        print("✅ dist构建目录存在")
        build_ok = True
    else:
        print("⚠️ dist构建目录不存在")
        build_ok = False
    
    # 提供解决方案
    print("\n🎯 推荐操作:")
    
    if not dependencies_ok:
        print("   1. 进入frontend目录: cd frontend")
        print("   2. 安装依赖: npm install")
    
    if not build_ok:
        print("   3. 构建项目: npm run build")
    
    print("   4. 启动开发服务器: npm run dev")
    print("   5. 或使用交互式启动器: python mujica_interactive.py")
    
    return node_ok and npm_ok and dependencies_ok

def quick_install():
    """快速安装前端依赖"""
    print("\n🚀 快速安装前端依赖...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ frontend目录不存在")
        return False
    
    try:
        print("📦 正在安装npm依赖...")
        result = subprocess.run(["npm", "install"], 
                              cwd=frontend_dir, check=True)
        print("✅ 依赖安装成功")
        
        print("🏗️ 正在构建项目...")
        result = subprocess.run(["npm", "run", "build"], 
                              cwd=frontend_dir, check=True)
        print("✅ 项目构建成功")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ npm命令未找到，请先安装Node.js")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="🔍 前端环境检查工具")
    parser.add_argument("--install", action="store_true", help="自动安装依赖")
    
    args = parser.parse_args()
    
    if args.install:
        quick_install()
    else:
        check_frontend_setup()

if __name__ == "__main__":
    main() 