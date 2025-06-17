#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML转PDF工具
将demo_webpage.html转换为PDF文档

依赖库：
- weasyprint: 用于HTML到PDF的转换
- 安装命令: pip install weasyprint

作者：AI助手
日期：2024-06-10
"""

import os
import sys
from pathlib import Path

def install_weasyprint():
    """安装weasyprint库"""
    try:
        import subprocess
        print("正在安装weasyprint库...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "weasyprint"])
        print("weasyprint安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装weasyprint失败: {e}")
        return False

def html_to_pdf(html_file_path, pdf_file_path, css_adjustments=None):
    """
    将HTML文件转换为PDF
    
    Args:
        html_file_path (str): HTML文件路径
        pdf_file_path (str): 输出PDF文件路径
        css_adjustments (str): 额外的CSS调整
    """
    try:
        # 尝试导入weasyprint
        try:
            import weasyprint
            from weasyprint import HTML, CSS
            print("weasyprint库已就绪")
        except ImportError:
            print("weasyprint库未安装，正在安装...")
            if install_weasyprint():
                import weasyprint
                from weasyprint import HTML, CSS
            else:
                print("无法安装weasyprint，请手动安装：pip install weasyprint")
                return False

        # 检查HTML文件是否存在
        if not os.path.exists(html_file_path):
            print(f"错误：HTML文件不存在 - {html_file_path}")
            return False

        print(f"正在转换: {html_file_path} -> {pdf_file_path}")

        # PDF专用的CSS调整
        pdf_css = """
        @page {
            size: A4;
            margin: 1cm;
        }
        
        body {
            font-size: 12px;
            line-height: 1.4;
        }
        
        .container {
            width: 100% !important;
            max-width: none !important;
            margin: 0 !important;
            padding: 10px !important;
            background: white !important;
            box-shadow: none !important;
            border-radius: 0 !important;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        
        .nav {
            background: #2c3e50 !important;
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        
        .section {
            page-break-inside: avoid;
            margin: 15px 0 !important;
            padding: 20px !important;
        }
        
        .feature-grid,
        .value-cards {
            display: grid !important;
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 15px !important;
        }
        
        .feature-card,
        .value-card {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            print-color-adjust: exact !important;
            page-break-inside: avoid;
        }
        
        .stats-table {
            font-size: 10px !important;
            page-break-inside: avoid;
        }
        
        .stats-table th {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        
        .highlight-box {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            print-color-adjust: exact !important;
            page-break-inside: avoid;
        }
        
        .formula-box {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            print-color-adjust: exact !important;
            page-break-inside: avoid;
        }
        
        .innovation-list {
            display: block !important;
        }
        
        .innovation-item {
            page-break-inside: avoid;
            margin-bottom: 15px !important;
        }
        
        .command {
            font-size: 10px !important;
            page-break-inside: avoid;
        }
        
        .footer {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        
        /* 隐藏滚动条和交互元素 */
        .scroll-top {
            display: none !important;
        }
        
        /* 确保图标显示 */
        .icon {
            font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji";
        }
        """

        # 如果有额外的CSS调整，添加到PDF CSS中
        if css_adjustments:
            pdf_css += css_adjustments

        # 读取HTML文件内容
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 创建HTML对象
        html_doc = HTML(string=html_content, base_url=os.path.dirname(html_file_path))
        
        # 创建CSS对象
        css_doc = CSS(string=pdf_css)

        # 生成PDF
        html_doc.write_pdf(pdf_file_path, stylesheets=[css_doc])
        
        print(f"✅ PDF转换成功！")
        print(f"📄 输出文件: {pdf_file_path}")
        
        # 显示文件大小
        pdf_size = os.path.getsize(pdf_file_path)
        print(f"📊 文件大小: {pdf_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def main():
    """主函数"""
    print("🔄 HTML转PDF转换工具")
    print("=" * 50)
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 输入和输出文件路径
    html_file = project_root / "docs" / "demo_webpage.html"
    pdf_file = project_root / "docs" / "基于多源数据融合的电网概率负荷预测方法_技术演示.pdf"
    
    print(f"📂 HTML文件: {html_file}")
    print(f"📄 PDF文件: {pdf_file}")
    print()
    
    # 执行转换
    success = html_to_pdf(str(html_file), str(pdf_file))
    
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
        print("\n❌ 转换失败，请检查错误信息")
        
        # 提供备用方案
        print("\n💡 备用方案：")
        print("1. 打开demo_webpage.html文件")
        print("2. 按Ctrl+P打开浏览器打印对话框")
        print("3. 选择'保存为PDF'")
        print("4. 调整打印设置（建议选择A4纸张，去掉页眉页脚）")
        print("5. 点击保存")

if __name__ == "__main__":
    main() 