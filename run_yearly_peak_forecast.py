#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
年度预测生成脚本 - 扩展版

此脚本用于生成整年的预测数据，支持多种预测类型：
- 日前预测 (day-ahead)
- 滚动预测 (rolling)
- 区间预测 (interval)
- 概率预测 (probabilistic)

它会按月调用scene_forecasting.py脚本，并将所有结果合并为CSV文件。
"""

import os
import sys
import subprocess
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse
import shutil
import calendar
import glob
import numpy as np

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def run_forecast_for_month(year, month, province, forecast_type="load", 
                          prediction_type="day-ahead", output_dir=None, 
                          interval=15, confidence_level=0.9, quantiles="0.1,0.5,0.9"):
    """
    为指定月份运行预测
    
    参数:
        year: 年份
        month: 月份 (1-12)
        province: 省份名称
        forecast_type: 预测类型 (load/pv/wind)
        prediction_type: 预测方式 (day-ahead/rolling/interval/probabilistic)
        output_dir: 输出目录
        interval: 预测间隔（分钟）
        confidence_level: 区间预测的置信水平
        quantiles: 概率预测的分位数
    
    返回:
        预测结果文件路径
    """
    # 构建输出目录
    if output_dir is None:
        # 根据预测类型选择输出路径
        if prediction_type == "day-ahead":
            output_dir = f"results/day_ahead/{forecast_type}/{province}"
        elif prediction_type == "rolling":
            output_dir = f"results/rolling/{forecast_type}/{province}"
        elif prediction_type == "interval":
            output_dir = f"results/interval_forecast/{forecast_type}/{province}"
        elif prediction_type == "probabilistic":
            output_dir = f"results/probabilistic/{forecast_type}/{province}"
        else:
            output_dir = f"results/{prediction_type}/{forecast_type}/{province}"
    
    ensure_dir(output_dir)
    
    # 计算月份的第一天和最后一天
    first_day = datetime(year, month, 1)
    _, last_day_num = calendar.monthrange(year, month)
    last_day = datetime(year, month, last_day_num)
    
    # 从1月2日开始（需要历史数据）
    if year == first_day.year and month == 1:
        first_day = datetime(year, 1, 2)
    
    first_day_str = first_day.strftime('%Y-%m-%d')
    last_day_str = last_day.strftime('%Y-%m-%d')
    
    # 构建输出文件名
    date_str = f"{year}-{month:02d}"
    if prediction_type == "interval":
        output_json = os.path.join(output_dir, f"interval_{forecast_type}_{province}_{date_str}.json")
    else:
        output_json = os.path.join(output_dir, f"forecast_{forecast_type}_{date_str}.json")
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/scene_forecasting.py",
        f"--forecast_type={forecast_type}",
        f"--province={province}",
        f"--interval={interval}",
        "--peak_aware",
        f"--output_json={output_json}"
    ]
    
    # 根据预测类型添加不同的参数
    if prediction_type == "day-ahead":
        cmd.extend([
            "--day_ahead",
            f"--forecast_date={first_day_str}",
            f"--forecast_end_date={last_day_str}",
            "--enhanced_smoothing"
        ])
    elif prediction_type == "rolling":
        cmd.extend([
            f"--forecast_start={first_day_str}",
            f"--forecast_end={last_day_str}",
            "--mode=forecast"
        ])
    elif prediction_type == "interval":
        cmd.extend([
            "--train_prediction_type=interval",
            "--day_ahead",
            f"--forecast_date={first_day_str}",
            f"--forecast_end_date={last_day_str}",
            f"--confidence_level={confidence_level}"
        ])
    elif prediction_type == "probabilistic":
        cmd.extend([
            "--train_prediction_type=probabilistic",
            "--day_ahead",
            f"--forecast_date={first_day_str}",
            f"--forecast_end_date={last_day_str}",
            f"--quantiles={quantiles}"
        ])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 运行命令
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 实时输出日志
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 等待进程完成
        return_code = process.poll()
        if return_code != 0:
            print(f"警告: 命令执行返回非零状态码: {return_code}")
        
        # 检查输出文件是否存在
        if not os.path.exists(output_json):
            print(f"错误: 预测结果文件未生成: {output_json}")
            return None
            
        return output_json
    
    except Exception as e:
        print(f"执行命令时出错: {e}")
        return None

def merge_monthly_forecasts(json_files_info, output_csv, province, 
                           forecast_type="load", prediction_type="day-ahead"):
    """
    合并所有月度预测结果为一个CSV文件
    
    参数:
        json_files_info: JSON文件路径列表
        output_csv: 输出CSV文件路径
        province: 省份名称
        forecast_type: 预测类型
        prediction_type: 预测方式
    """
    print(f"开始合并{province}的{forecast_type} {prediction_type}预测结果...")
    
    all_data = []
    
    if not json_files_info:
        print(f"错误: 没有可用的JSON文件进行合并")
        return False
    
    print(f"找到{len(json_files_info)}个JSON文件")
    
    # 读取并合并所有JSON文件
    for json_file in json_files_info:
        if not os.path.exists(json_file):
            print(f"警告: 文件不存在 {json_file}")
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'predictions' in data and isinstance(data['predictions'], list):
                # 提取预测数据
                df = pd.DataFrame(data['predictions'])
                if not df.empty:
                    # 根据预测类型处理数据
                    if prediction_type == "interval":
                        # 确保区间预测包含必要的列
                        required_cols = ['datetime', 'predicted', 'lower_bound', 'upper_bound']
                        if all(col in df.columns for col in required_cols):
                            all_data.append(df)
                            print(f"从 {os.path.basename(json_file)} 中读取了 {len(df)} 条区间预测记录")
                        else:
                            print(f"警告: {json_file} 缺少区间预测必需的列")
                    elif prediction_type == "probabilistic":
                        # 概率预测应包含多个分位数列
                        if 'predicted' in df.columns:
                            all_data.append(df)
                            print(f"从 {os.path.basename(json_file)} 中读取了 {len(df)} 条概率预测记录")
                    else:
                        # 标准预测
                        if 'predicted' in df.columns:
                            all_data.append(df)
                            print(f"从 {os.path.basename(json_file)} 中读取了 {len(df)} 条记录")
            else:
                print(f"警告: {json_file}中未找到有效的预测数据")
        
        except Exception as e:
            print(f"读取{json_file}时出错: {e}")
    
    if not all_data:
        print("错误: 未能从任何JSON文件中提取有效数据")
        return False
    
    # 合并所有数据框
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 确保datetime列存在并格式化
    if 'datetime' in merged_df.columns:
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
        # 按时间排序
        merged_df = merged_df.sort_values('datetime')
        # 删除重复的时间点
        merged_df = merged_df.drop_duplicates(subset=['datetime'], keep='first')
    
    # 添加元数据列
    merged_df['province'] = province
    merged_df['forecast_type'] = forecast_type
    merged_df['prediction_type'] = prediction_type
    
    # 保存为CSV
    ensure_dir(os.path.dirname(output_csv))
    merged_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"已将合并结果保存至: {output_csv}")
    print(f"数据行数: {len(merged_df)}")
    print(f"数据列: {merged_df.columns.tolist()}")
    
    # 输出数据统计信息
    if 'datetime' in merged_df.columns:
        print(f"时间范围: {merged_df['datetime'].min()} 至 {merged_df['datetime'].max()}")
    
    return True

def generate_year_forecast(year, province, forecast_type="load", 
                          prediction_type="day-ahead", interval=15,
                          confidence_level=0.9, quantiles="0.1,0.5,0.9"):
    """
    生成指定年份的全年预测
    
    参数:
        year: 年份
        province: 省份名称
        forecast_type: 预测类型
        prediction_type: 预测方式
        interval: 预测间隔（分钟）
        confidence_level: 区间预测的置信水平
        quantiles: 概率预测的分位数
    """
    # 记录开始时间
    start_time = time.time()
    
    # 存储成功生成的JSON文件路径
    json_files = []
    
    # 遍历每一个月
    successful_months = 0
    failed_months = 0
    
    for month in range(1, 13):
        print(f"\n===== 开始处理 {year}年{month}月 的{prediction_type}预测 =====")
        
        # 运行预测
        output_json = run_forecast_for_month(
            year=year,
            month=month,
            province=province,
            forecast_type=forecast_type,
            prediction_type=prediction_type,
            interval=interval,
            confidence_level=confidence_level,
            quantiles=quantiles
        )
        
        if output_json:
            successful_months += 1
            json_files.append(output_json)
        else:
            failed_months += 1
        
        # 每处理3个月，输出进度
        if (successful_months + failed_months) % 3 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_month = elapsed_time / (successful_months + failed_months)
            remaining_months = 12 - (successful_months + failed_months)
            estimated_remaining_time = avg_time_per_month * remaining_months
            
            print(f"\n进度: 已处理 {successful_months + failed_months}/12 个月")
            print(f"成功: {successful_months}, 失败: {failed_months}")
            print(f"已用时间: {elapsed_time/60:.2f} 分钟")
            print(f"预计剩余时间: {estimated_remaining_time/60:.2f} 分钟")
    
    # 合并所有月度预测结果
    output_csv = f"results/yearly_forecasts/{forecast_type}_{province}_{year}_{prediction_type}.csv"
    ensure_dir("results/yearly_forecasts")
    
    merge_success = merge_monthly_forecasts(
        json_files, output_csv, province, forecast_type, prediction_type
    )
    
    # 输出总结
    total_time = time.time() - start_time
    print("\n===== 预测完成 =====")
    print(f"预测类型: {prediction_type}")
    print(f"总月数: 12")
    print(f"成功月数: {successful_months}")
    print(f"失败月数: {failed_months}")
    print(f"总用时: {total_time/60:.2f} 分钟")
    
    if merge_success:
        print(f"合并后的CSV文件: {output_csv}")
    
    return output_csv if merge_success else None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成整年的预测数据（支持多种预测类型）')
    parser.add_argument('--year', type=int, default=2024,
                      help='要预测的年份 (默认: 2024)')
    parser.add_argument('--province', type=str, default='浙江',
                      help='省份名称 (默认: 上海)')
    parser.add_argument('--forecast_type', type=str, choices=['load', 'pv', 'wind'], default='load',
                      help='预测类型: load=负荷, pv=光伏, wind=风电 (默认: load)')
    parser.add_argument('--prediction_type', type=str, 
                      choices=['day-ahead', 'rolling', 'interval', 'probabilistic'], 
                      default='rolling',
                      help='预测方式 (默认: day-ahead)')
    parser.add_argument('--interval', type=int, default=15,
                      help='预测间隔（分钟）(默认: 15)')
    parser.add_argument('--confidence_level', type=float, default=0.9,
                      help='区间预测的置信水平 (默认: 0.9)')
    parser.add_argument('--quantiles', type=str, default='0.1,0.5,0.9',
                      help='概率预测的分位数 (默认: 0.1,0.5,0.9)')
    parser.add_argument('--clear_previous', action='store_true', default=False,
                      help='是否清除之前的预测结果 (默认: False)')
    parser.add_argument('--month', type=int, default=None, 
                      help='仅预测指定月份 (1-12)，不指定则预测全年')
    parser.add_argument('--all_types', action='store_true', default=False,
                      help='生成所有类型的预测 (默认: False)')
    
    args = parser.parse_args()
    
    # 如果需要生成所有类型的预测
    if args.all_types:
        prediction_types = ['day-ahead', 'rolling', 'interval']
        for pred_type in prediction_types:
            print(f"\n{'='*60}")
            print(f"开始生成 {pred_type} 类型的年度预测")
            print(f"{'='*60}")
            
            output_csv = generate_year_forecast(
                year=args.year,
                province=args.province,
                forecast_type=args.forecast_type,
                prediction_type=pred_type,
                interval=args.interval,
                confidence_level=args.confidence_level,
                quantiles=args.quantiles
            )
            
            if output_csv:
                print(f"{pred_type} 预测成功完成")
            else:
                print(f"{pred_type} 预测过程中出现错误")
            
            # 每种类型之间暂停一下
            time.sleep(5)
    else:
        # 单一类型预测
        if args.clear_previous:
            # 清除之前的结果
            if args.prediction_type == "day-ahead":
                json_dir = f"results/day_ahead/{args.forecast_type}/{args.province}"
            elif args.prediction_type == "rolling":
                json_dir = f"results/rolling/{args.forecast_type}/{args.province}"
            elif args.prediction_type == "interval":
                json_dir = f"results/interval_forecast/{args.forecast_type}/{args.province}"
            elif args.prediction_type == "probabilistic":
                json_dir = f"results/probabilistic/{args.forecast_type}/{args.province}"
            else:
                json_dir = f"results/{args.prediction_type}/{args.forecast_type}/{args.province}"
                
            if os.path.exists(json_dir):
                print(f"清除之前的预测结果: {json_dir}")
                files_to_delete = glob.glob(f"{json_dir}/*.json")
                for file in files_to_delete:
                    os.remove(file)
                    print(f"删除文件: {file}")
        
        # 如果指定了单月预测
        if args.month is not None:
            if args.month < 1 or args.month > 12:
                print(f"错误: 月份必须是1-12之间的整数，获得了: {args.month}")
                return
                
            print(f"仅预测 {args.year}年{args.month}月 的{args.prediction_type}数据")
            
            output_json = run_forecast_for_month(
                year=args.year,
                month=args.month,
                province=args.province,
                forecast_type=args.forecast_type,
                prediction_type=args.prediction_type,
                interval=args.interval,
                confidence_level=args.confidence_level,
                quantiles=args.quantiles
            )
            
            if output_json:
                print(f"预测成功，结果已保存至: {output_json}")
            else:
                print("预测失败，请检查日志")
        else:
            # 生成整年预测
            output_csv = generate_year_forecast(
                year=args.year,
                province=args.province,
                forecast_type=args.forecast_type,
                prediction_type=args.prediction_type,
                interval=args.interval,
                confidence_level=args.confidence_level,
                quantiles=args.quantiles
            )
            
            if output_csv:
                print(f"预测成功完成，结果已保存至: {output_csv}")
            else:
                print("预测过程中出现错误，请检查日志")

if __name__ == '__main__':
    main()