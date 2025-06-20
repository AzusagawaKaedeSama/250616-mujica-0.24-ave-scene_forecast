#!/usr/bin/env python3
"""
🌐 MUJICA 前端启动器
支持多种前端技术栈的启动方式
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import subprocess
import time
import platform
from pathlib import Path

class FrontendServer:
    """前端开发服务器"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.directory = Path(__file__).parent
        
    def get_npm_cmd(self):
        """获取正确的npm命令（Windows兼容性）"""
        if platform.system() == "Windows":
            return "npm.cmd"
        return "npm"
    
    def check_npm_available(self):
        """检查npm是否可用"""
        npm_cmd = self.get_npm_cmd()
        try:
            result = subprocess.run([npm_cmd, "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Node.js环境检测成功 (npm {result.stdout.strip()})")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ 未找到npm命令")
            print("💡 请确保已安装Node.js并将其添加到系统PATH")
            print("   下载地址: https://nodejs.org/")
            return False
        
    def start_vite_dev_server(self, open_browser: bool = True):
        """启动Vite开发服务器（推荐）"""
        print("🚀 尝试启动Vite开发服务器...")
        
        # 先检查npm是否可用
        if not self.check_npm_available():
            return False
        
        # 切换到前端目录
        os.chdir(self.directory)
        
        # 检查node_modules
        if not (self.directory / "node_modules").exists():
            print("⚠️ 未找到node_modules，正在安装依赖...")
            npm_cmd = self.get_npm_cmd()
            try:
                subprocess.run([npm_cmd, "install"], check=True)
                print("✅ 依赖安装完成")
            except subprocess.CalledProcessError:
                print("❌ 依赖安装失败，请手动运行: npm install")
                return False
        
        try:
            # 启动Vite开发服务器
            print("✅ 启动Vite开发服务器...")
            print(f"   React应用: http://localhost:5173")
            print(f"   API代理: http://localhost:5173/api -> http://localhost:5001")
            print("   支持热重载和完整React功能")
            print("   按 Ctrl+C 停止服务器\n")
            
            if open_browser:
                # 延迟打开浏览器
                def open_browser_delayed():
                    time.sleep(3)
                    webbrowser.open("http://localhost:5173")
                
                import threading
                threading.Thread(target=open_browser_delayed, daemon=True).start()
            
            # 运行Vite开发服务器
            npm_cmd = self.get_npm_cmd()
            result = subprocess.run([npm_cmd, "run", "dev"], check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Vite开发服务器启动失败: {e}")
            return False
        except KeyboardInterrupt:
            print("\n👋 Vite开发服务器已停止")
            return True
    
    def serve_dist_directory(self, open_browser: bool = True):
        """服务构建后的dist目录"""
        print("📦 尝试服务构建后的应用...")
        
        os.chdir(self.directory)
        dist_dir = self.directory / "dist"
        
        if not dist_dir.exists():
            print("⚠️ 未找到dist目录，尝试构建...")
            if self.check_npm_available():
                npm_cmd = self.get_npm_cmd()
                try:
                    subprocess.run([npm_cmd, "run", "build"], check=True)
                    print("✅ 构建完成")
                except subprocess.CalledProcessError:
                    print("❌ 构建失败，请手动运行: npm run build")
                    return False
            else:
                return False
        
        try:
            os.chdir(dist_dir)
            url = f"http://localhost:{self.port}"
            print(f"✅ 服务构建后的应用: {url}")
            print("   按 Ctrl+C 停止服务器\n")
            
            if open_browser:
                print(f"🔗 自动打开浏览器: {url}")
                webbrowser.open(url)
            
            with socketserver.TCPServer(("", self.port), http.server.SimpleHTTPRequestHandler) as httpd:
                httpd.serve_forever()
            return True
            
        except Exception as e:
            print(f"❌ 服务构建应用失败: {e}")
            return False
    
    def start_simple_server(self, open_browser: bool = True):
        """启动简单HTTP服务器（备用方案）"""
        print("🔧 启动简单HTTP服务器（仅支持静态文件）...")
        
        os.chdir(self.directory)
        
        try:
            with socketserver.TCPServer(("", self.port), http.server.SimpleHTTPRequestHandler) as httpd:
                url = f"http://localhost:{self.port}"
                print(f"✅ 简单HTTP服务器启动: {url}")
                print("   可用页面:")
                print("   - dashboard.html  - 静态预测仪表板（推荐）")
                print("   - index.html      - React应用（可能有兼容性问题）")
                print("   按 Ctrl+C 停止服务器\n")
                
                if open_browser:
                    print(f"🔗 自动打开仪表板: {url}/dashboard.html")
                    webbrowser.open(f"{url}/dashboard.html")
                
                httpd.serve_forever()
                
        except KeyboardInterrupt:
            print("\n👋 服务器已停止")
        except Exception as e:
            print(f"❌ 服务器启动失败: {e}")
            
    def start(self, open_browser: bool = True, force_simple: bool = False):
        """智能启动前端服务器"""
        print("🌐 MUJICA 前端服务器启动中...")
        print("="*50)
        
        if force_simple:
            self.start_simple_server(open_browser)
            return
        
        # 尝试启动顺序：Vite开发服务器 -> 构建版本 -> 简单服务器
        print("🎯 智能选择最佳启动方式...\n")
        
        # 方式1：Vite开发服务器（最佳体验）
        if self.start_vite_dev_server(open_browser):
            return
        
        print("\n" + "="*50)
        print("🔄 尝试备用方案...\n")
        
        # 方式2：服务构建后的版本
        if self.serve_dist_directory(open_browser):
            return
        
        print("\n" + "="*50)
        print("🔄 使用最后备用方案...\n")
        
        # 方式3：简单HTTP服务器
        print("💡 提示：建议使用 dashboard.html 获得最佳体验")
        self.start_simple_server(open_browser)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="🌐 MUJICA 前端开发服务器")
    parser.add_argument("--port", type=int, default=8080, help="端口号 (默认: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--simple", action="store_true", help="强制使用简单HTTP服务器")
    parser.add_argument("--vite-only", action="store_true", help="仅尝试Vite开发服务器")
    
    args = parser.parse_args()
    
    server = FrontendServer(port=args.port)
    
    if args.vite_only:
        server.start_vite_dev_server(open_browser=not args.no_browser)
    else:
        server.start(open_browser=not args.no_browser, force_simple=args.simple)

if __name__ == "__main__":
    main() 