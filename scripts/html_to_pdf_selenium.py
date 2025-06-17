#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML转PDF工具 (使用Selenium + Chrome)
将demo_webpage.html转换为PDF文档

依赖库：
- selenium: 用于浏览器自动化
- webdriver-manager: 自动管理Chrome驱动
- 安装命令: pip install selenium webdriver-manager

作者：AI助手
日期：2024-06-10
"""

import os
import sys
import time
import json
from pathlib import Path

def install_selenium():
    """安装selenium相关库"""
    try:
        import subprocess
        print("正在安装selenium和webdriver-manager...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium", "webdriver-manager"])
        print("selenium安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装selenium失败: {e}")
        return False

def html_to_pdf_selenium(html_file_path, pdf_file_path):
    """
    使用Selenium + Chrome将HTML文件转换为PDF
    
    Args:
        html_file_path (str): HTML文件路径
        pdf_file_path (str): 输出PDF文件路径
    """
    try:
        # 尝试导入selenium
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
            print("selenium库已就绪")
        except ImportError:
            print("selenium库未安装，正在安装...")
            if install_selenium():
                from selenium import webdriver
                from selenium.webdriver.chrome.service import Service
                from selenium.webdriver.chrome.options import Options
                from webdriver_manager.chrome import ChromeDriverManager
            else:
                print("无法安装selenium，请手动安装：pip install selenium webdriver-manager")
                return False

        # 检查HTML文件是否存在
        if not os.path.exists(html_file_path):
            print(f"错误：HTML文件不存在 - {html_file_path}")
            return False

        print(f"正在转换: {html_file_path} -> {pdf_file_path}")

        # 设置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # PDF打印设置
        pdf_print_options = {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'preferCSSPageSize': True,
            'paperWidth': 8.27,  # A4 width in inches
            'paperHeight': 11.7,  # A4 height in inches
            'marginTop': 0.4,
            'marginBottom': 0.4,
            'marginLeft': 0.4,
            'marginRight': 0.4,
        }

        # 启动Chrome浏览器
        print("正在启动Chrome浏览器...")
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            print(f"启动Chrome失败: {e}")
            print("请确保已安装Chrome浏览器")
            return False

        try:
            # 转换为文件URL
            file_url = f"file:///{os.path.abspath(html_file_path).replace(os.sep, '/')}"
            print(f"正在加载页面: {file_url}")
            
            # 加载HTML页面
            driver.get(file_url)
            
            # 等待页面完全加载
            time.sleep(3)
            
            # 执行JavaScript等待所有图片和CSS加载完成
            driver.execute_script("""
                return new Promise((resolve) => {
                    if (document.readyState === 'complete') {
                        setTimeout(resolve, 1000);
                    } else {
                        window.addEventListener('load', () => {
                            setTimeout(resolve, 1000);
                        });
                    }
                });
            """)
            
            print("页面加载完成，开始生成PDF...")
            
            # 使用Chrome DevTools Protocol生成PDF
            result = driver.execute_cdp_cmd("Page.printToPDF", pdf_print_options)
            
            # 获取PDF数据并保存
            pdf_data = result['data']
            
            # 将base64数据解码并保存为PDF文件
            import base64
            with open(pdf_file_path, 'wb') as f:
                f.write(base64.b64decode(pdf_data))
            
            print(f"✅ PDF转换成功！")
            print(f"📄 输出文件: {pdf_file_path}")
            
            # 显示文件大小
            pdf_size = os.path.getsize(pdf_file_path)
            print(f"📊 文件大小: {pdf_size / 1024 / 1024:.2f} MB")
            
            return True
            
        finally:
            # 关闭浏览器
            driver.quit()
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def create_print_friendly_html(original_html_path, print_html_path):
    """
    创建一个打印友好的HTML版本
    """
    try:
        # 读取原始HTML
        with open(original_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 添加打印专用CSS
        print_css = """
        <style media="print">
        @page {
            size: A4;
            margin: 1cm;
        }
        
        body {
            font-size: 11px !important;
            line-height: 1.3 !important;
            background: white !important;
        }
        
        .container {
            width: 100% !important;
            max-width: none !important;
            margin: 0 !important;
            padding: 5px !important;
            background: white !important;
            box-shadow: none !important;
            border-radius: 0 !important;
        }
        
        .header {
            background: #1e3c72 !important;
            color: white !important;
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
        }
        
        .nav {
            background: #2c3e50 !important;
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
        }
        
        .section {
            page-break-inside: avoid;
            margin: 10px 0 !important;
            padding: 15px !important;
        }
        
        .feature-grid {
            display: grid !important;
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 10px !important;
        }
        
        .feature-card {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            page-break-inside: avoid;
            font-size: 10px !important;
        }
        
        .stats-table {
            font-size: 9px !important;
            page-break-inside: avoid;
        }
        
        .stats-table th {
            background: #2c3e50 !important;
            color: white !important;
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
        }
        
        .highlight-box {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            page-break-inside: avoid;
        }
        
        .value-cards {
            display: grid !important;
            grid-template-columns: repeat(4, 1fr) !important;
            gap: 8px !important;
        }
        
        .value-card {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            page-break-inside: avoid;
            font-size: 9px !important;
        }
        
        .formula-box {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            page-break-inside: avoid;
            font-size: 10px !important;
        }
        
        .innovation-item {
            page-break-inside: avoid;
            margin-bottom: 10px !important;
        }
        
        .command {
            font-size: 8px !important;
            page-break-inside: avoid;
        }
        
        .footer {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
        }
        
        .scroll-top {
            display: none !important;
        }
        </style>
        """
        
        # 在</head>标签前插入打印CSS
        html_content = html_content.replace('</head>', print_css + '</head>')
        
        # 保存打印版本
        with open(print_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return True
        
    except Exception as e:
        print(f"创建打印友好版本失败: {e}")
        return False

def main():
    """主函数"""
    print("🔄 HTML转PDF转换工具 (Selenium版)")
    print("=" * 50)
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 输入和输出文件路径
    html_file = project_root / "docs" / "demo_webpage.html"
    print_html_file = project_root / "docs" / "demo_webpage_print.html"
    pdf_file = project_root / "docs" / "基于多源数据融合的电网概率负荷预测方法_技术演示.pdf"
    
    print(f"📂 HTML文件: {html_file}")
    print(f"📄 PDF文件: {pdf_file}")
    print()
    
    # 创建打印友好的HTML版本
    print("📝 创建打印友好版本...")
    if not create_print_friendly_html(str(html_file), str(print_html_file)):
        print("❌ 创建打印版本失败")
        return
    
    # 执行转换
    success = html_to_pdf_selenium(str(print_html_file), str(pdf_file))
    
    # 清理临时文件
    try:
        if print_html_file.exists():
            print_html_file.unlink()
    except:
        pass
    
    if success:
        print("\n🎉 转换完成！")
        print(f"您可以在以下位置找到PDF文件：")
        print(f"{pdf_file}")
        
        # 尝试打开PDF文件所在目录
        try:
            import subprocess
            import platform
            
            pdf_dir = pdf_file.parent
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(pdf_dir)], check=False)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(pdf_dir)], check=False)
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", str(pdf_dir)], check=False)
        except:
            pass
            
    else:
        print("\n❌ 转换失败，请尝试以下备用方案：")
        print("\n💡 备用方案1 - 浏览器打印：")
        print("1. 打开demo_webpage.html文件")
        print("2. 按Ctrl+P打开浏览器打印对话框")
        print("3. 选择'保存为PDF'")
        print("4. 调整打印设置（建议选择A4纸张，启用背景图形）")
        print("5. 点击保存")
        
        print("\n💡 备用方案2 - 在线转换：")
        print("1. 访问在线HTML转PDF网站（如：https://www.ilovepdf.com/html-to-pdf）")
        print("2. 上传demo_webpage.html文件")
        print("3. 下载生成的PDF文件")

if __name__ == "__main__":
    main() 