# 改进的plot_style.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform
import matplotlib as mpl
import sys
from pathlib import Path


def find_font_file(font_name):
    """查找字体文件的完整路径"""
    # 获取所有可用字体的详细信息
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    
    # 转换为小写以进行不区分大小写的搜索
    font_name_lower = font_name.lower()
    
    # 遍历所有字体文件,查找包含font_name的文件
    matches = []
    for font_file in font_files:
        if font_name_lower in os.path.basename(font_file).lower():
            matches.append(font_file)
    
    if matches:
        # print(f"找到'{font_name}'相关的字体文件: {matches}")
        return matches
    else:
        # print(f"未找到'{font_name}'相关的字体文件")
        return []

def clear_font_cache():
    """清除matplotlib的字体缓存"""
    cache_dir = Path(mpl.get_cachedir())
    font_cache = list(cache_dir.glob('*font*cache*'))
    
    if font_cache:
        for cache in font_cache:
            try:
                cache.unlink()  # 删除缓存文件
                print(f"已删除字体缓存: {cache}")
            except Exception as e:
                print(f"无法删除缓存文件 {cache}: {e}")
    # else:
    #     print("未找到字体缓存文件")

def apply_style(force=True):
    """强制应用样式设置，避免被其他设置覆盖"""
    # 首先查找Times New Roman字体
    times_fonts = find_font_file("times")
    
    # 注册找到的Times New Roman字体
    times_font_added = False
    if times_fonts:
        for font_path in times_fonts:
            try:
                fm.fontManager.addfont(font_path)
                times_font_added = True
            except Exception as e:
                print(f"添加字体'{font_path}'时出错: {e}")
    
    # 查找中文字体
    chinese_fonts = ["SimHei", "Microsoft YaHei", "SimSun"]
    
    # 检查系统中的中文字体
    for font in chinese_fonts:
        chinese_font_files = find_font_file(font)
        if chinese_font_files:
            for font_path in chinese_font_files:
                try:
                    fm.fontManager.addfont(font_path)
                    # print(f"已添加中文字体: {font}")
                except Exception as e:
                    print(f"添加中文字体'{font}'时出错: {e}")
    
    # 如果发现了Times字体并添加成功，则刷新字体缓存
    if times_font_added:
        clear_font_cache()
        # 重新构建字体列表

    # 确保Times New Roman的有效性
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Times New Roman' in all_fonts:
        # print("Times New Roman字体已成功注册并可用")
        chinese_fonts = chinese_fonts
    else:
        print(f"警告: Times New Roman字体不可用! 可用的字体有: {all_fonts[:10]}...")
    
    # 设置字体顺序 - 确保Times New Roman在第一位
    plt.rcParams['font.family'] = ['serif', 'sans-serif']  # 优先使用serif字体族
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 显式设置数学字体
    plt.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern数学字体
    plt.rcParams['mathtext.rm'] = 'serif'  # 对数学公式中的普通文本使用serif字体
    
    # 设置全局样式参数
    plt.rcParams['figure.figsize'] = (14, 8)  # 默认图表大小
    plt.rcParams['figure.dpi'] = 100  # 图表DPI
    
    # 线条和标记样式
    plt.rcParams['lines.linewidth'] = 2.0  # 线条宽度
    plt.rcParams['lines.markersize'] = 8   # 标记大小
    
    # 轴样式
    plt.rcParams['axes.grid'] = True       # 默认显示网格
    plt.rcParams['grid.alpha'] = 0.3       # 网格透明度
    plt.rcParams['grid.linestyle'] = '--'  # 网格线型
    
    # 字体大小
    plt.rcParams['font.size'] = 12         # 基础字体大小
    plt.rcParams['axes.titlesize'] = 16    # 标题字体大小
    plt.rcParams['axes.labelsize'] = 14    # 轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 12   # x轴刻度标签大小
    plt.rcParams['ytick.labelsize'] = 12   # y轴刻度标签大小
    
    # 图例样式
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.framealpha'] = 0.7
    plt.rcParams['legend.edgecolor'] = 'gray'
    plt.rcParams['legend.fontsize'] = 12
    
    # 保存图片设置
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # 如果需要强制应用，设置一个标志
    if force:
        # 保存原始的plot函数
        original_plot = plt.plot
        
        # 重新定义plot函数，确保在每次使用时重新应用样式
        def styled_plot(*args, **kwargs):
            # 设置字体，确保每次绘图时使用
            plt.rcParams['font.family'] = ['serif', 'sans-serif']
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
            return original_plot(*args, **kwargs)
        
        # 替换plt.plot函数
        plt.plot = styled_plot
        
        # 同样处理figure函数
        original_figure = plt.figure
        
        def styled_figure(*args, **kwargs):
            # 设置字体，确保每次创建figure时使用
            plt.rcParams['font.family'] = ['serif', 'sans-serif']
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
            return original_figure(*args, **kwargs)
        
        plt.figure = styled_figure
        
        # print("样式已强制应用于所有图表函数")

def test_times_new_roman():
    """测试Times New Roman字体是否正确应用"""
    plt.figure(figsize=(10, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-', label='Sample Data')
    plt.title('Test Plot with Times New Roman Font')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.legend()
    plt.grid(True)
    
    # 添加测试文本
    plt.text(2, 8, 'This should be Times New Roman ', fontsize=14)
    
    # 显示当前使用的字体信息
    from matplotlib.font_manager import findfont, FontProperties
    font = findfont(FontProperties(family='serif'))
    plt.figtext(0.5, 0.01, f'Using font: {font}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"当前serif字体: {plt.rcParams['font.serif']}")
    print(f"当前sans-serif字体: {plt.rcParams['font.sans-serif']}")

# 在导入时自动应用样式
apply_style()

# 提供一个函数，显式重新应用样式
def reset_style():
    """重置并重新应用样式，用于覆盖被修改的样式设置"""
    apply_style(force=True)
    print("绘图样式已重置")

# 一些预定义的颜色方案
COLOR_SCHEMES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'pastel': ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a'],
    'dark': ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'],
    'colorblind': ['#0072B2', '#E69F00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#D55E00', '#000000']
}

# 提供一个函数来应用颜色方案
def use_color_scheme(scheme_name='default'):
    """应用预定义的颜色方案"""
    if scheme_name in COLOR_SCHEMES:
        plt.rcParams['axes.prop_cycle'] = plt.cycler('color', COLOR_SCHEMES[scheme_name])
        print(f"已应用'{scheme_name}'颜色方案")
    else:
        print(f"未找到颜色方案'{scheme_name}'，可用方案: {list(COLOR_SCHEMES.keys())}")

if __name__ == "__main__":
    # 如果直接运行此脚本，执行测试
    test_times_new_roman()

