import cdsapi
import os
import zipfile
import pandas as pd
import numpy as np
import time
from datetime import datetime
import shutil

def create_directory(directory):
    """创建目录，如果目录不存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def process_downloaded_file(file_path, extract_to):
    """处理下载的文件（可能是zip或直接是csv）"""
    # 获取文件扩展名
    _, ext = os.path.splitext(file_path)
    
    # 检查文件类型
    if ext.lower() == '.zip':
        try:
            # 尝试解压zip文件
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"成功解压文件 {file_path} 到 {extract_to}")
            return True
        except zipfile.BadZipFile:
            print(f"文件 {file_path} 不是有效的zip文件，将直接复制")
    
    # 如果不是zip或解压失败，直接复制文件到目标目录
    file_name = os.path.basename(file_path)
    target_path = os.path.join(extract_to, file_name)
    shutil.copy2(file_path, target_path)
    print(f"复制文件 {file_path} 到 {target_path}")
    return True

def get_era5_timeseries_data(longitude, latitude, year, month, output_dir):
    """获取指定经纬度点和月份的ERA5时间序列数据"""
    # 格式化坐标点信息用于文件名
    location_str = f"lon{longitude:.2f}_lat{latitude:.2f}"
    
    # 设置日期范围
    days_in_month = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30
    if month == 2:
        days_in_month = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{days_in_month}"
    date_range = f"{start_date}/{end_date}"
    date_str = f"{year}-{month:02d}"
    
    # 设置输出文件名
    output_filename = f"era5_data_{location_str}_{date_str}.zip"
    output_path = os.path.join(output_dir, "downloads", output_filename)
    
    # 确保下载目录存在
    create_directory(os.path.dirname(output_path))
    
    # 检查文件是否已存在
    if os.path.exists(output_path):
        print(f"文件已存在，跳过下载: {output_path}")
        return output_path
    
    # 准备API请求参数 - 使用时间序列API格式
    dataset = "reanalysis-era5-single-levels-timeseries"
    request = {
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind"
        ],
        "location": {
            "longitude": longitude,
            "latitude": latitude
        },
        "date": [date_range],
        "data_format": "csv"
    }
    
    # 发起API请求
    try:
        print(f"开始下载 {location_str} {date_str} 的数据...")
        client = cdsapi.Client()
        client.retrieve(dataset, request, output_path)
        print(f"下载完成: {output_path}")
        return output_path
    except Exception as e:
        print(f"下载失败: {str(e)}")
        return None

def process_all_grid_points(lon_min, lon_max, lat_min, lat_max, lon_step, lat_step, year, output_dir):
    """处理给定范围内的所有网格点"""
    # 创建输出目录
    create_directory(output_dir)
    extract_dir = os.path.join(output_dir, "extracted")
    create_directory(extract_dir)
    
    # 生成经纬度网格
    longitudes = np.arange(lon_min, lon_max + 0.5 * lon_step, lon_step)
    latitudes = np.arange(lat_min, lat_max + 0.5 * lat_step, lat_step)
    
    total_points = len(longitudes) * len(latitudes) * 12  # 12个月
    current_point = 0
    
    # 循环处理每个网格点和月份
    for lon in longitudes:
        for lat in latitudes:
            # 创建该点的主目录
            point_dir = os.path.join(extract_dir, f"lon{lon:.2f}_lat{lat:.2f}")
            create_directory(point_dir)
            
            for month in range(1, 13):  # 处理1-12月
                current_point += 1
                print(f"处理点 {current_point}/{total_points}: 经度 {lon:.2f}，纬度 {lat:.2f}，{year}年{month}月")
                
                # 创建月份目录
                month_dir = os.path.join(point_dir, f"{year}-{month:02d}")
                create_directory(month_dir)
                
                # 获取数据
                downloaded_file = get_era5_timeseries_data(lon, lat, year, month, output_dir)
                
                if downloaded_file and os.path.exists(downloaded_file):
                    # 处理下载的文件
                    process_downloaded_file(downloaded_file, month_dir)
                
                # 添加延迟以避免API请求过于频繁
                time.sleep(2)

def main():
    # 设置参数：经纬度范围、分辨率、年份、输出目录
    # 这里设置为华东地区的大致范围（可根据需要调整）
    lon_min, lon_max = 114.5, 122.5  # 经度范围
    lat_min, lat_max = 24.0, 35.0    # 纬度范围
    lon_step = 1                     # 经度步长（度）
    lat_step = 1                     # 纬度步长（度）
    year = 2024                      # 年份
    output_dir = "atmosphere_data"   # 输出目录
    
    print(f"准备获取区域 [{lon_min}°E-{lon_max}°E, {lat_min}°N-{lat_max}°N] 的数据")
    print(f"网格分辨率: {lon_step}° × {lat_step}°")
    print(f"年份: {year}")
    
    # 处理所有网格点
    process_all_grid_points(lon_min, lon_max, lat_min, lat_max, lon_step, lat_step, year, output_dir)
    
    print("所有数据处理完成！")
    print(f"下载的原始文件保存在: {os.path.join(output_dir, 'downloads')}")
    print(f"处理后的数据保存在: {os.path.join(output_dir, 'extracted')}")

if __name__ == "__main__":
    main()
