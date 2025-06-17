#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch-based load forecasting script using TorchConvTransformer model.
Modified to perform intraday rolling forecasts with optional training.
"""

import os
import sys

# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录 (scripts 目录)
current_script_dir = os.path.dirname(current_script_path)
# 获取项目根目录 (scripts 目录的上级目录)
project_root_dir = os.path.dirname(current_script_dir)

# 将项目根目录添加到 Python 搜索路径中
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
import pickle
import warnings
import argparse

warnings.filterwarnings('ignore')
import torch
import shutil
import json

# Import project components
# from data.data_loader import DataLoader
# from data.dataset_builder import DatasetBuilder
from utils.simple_evaluator import *

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def str_to_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """主函数，提供命令行参数解析并根据参数执行相应功能"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='负荷、光伏、风电预测工具')
    parser.add_argument('--data_path', type=str, default=None,
                      help='时间序列数据路径 (留空则根据类型和省份自动构建)')
    parser.add_argument('--forecast_type', type=str, choices=['load', 'pv', 'wind'], default='load',
                      help='预测类型: load=负荷, pv=光伏, wind=风电 (默认: load)')
    parser.add_argument('--mode', type=str, choices=['train', 'forecast', 'both'], default='forecast',
                      help='操作模式: train=仅训练, forecast=仅预测, both=训练并预测 (默认: forecast)')
    parser.add_argument('--retrain', action='store_true', default=False,
                      help='是否重新训练现有模型 (默认: False)')
    parser.add_argument('--train_start', type=str, default='2024-01-01',
                      help='训练数据开始日期 (默认: 2024-03-01)')
    parser.add_argument('--train_end', type=str, default='2024-03-31',
                      help='训练数据结束日期 (默认: 2024-03-30)')
    parser.add_argument('--forecast_start', type=str, default='2024-04-01',
                      help='预测开始日期 (默认: 2024-04-01)')
    parser.add_argument('--forecast_end', type=str, default='2024-04-01',
                      help='预测结束日期 (默认: 2024-04-07)')
    parser.add_argument('--interval', type=int, default=15,
                      help='预测间隔（分钟）(默认: 15)')
    parser.add_argument('--use_patterns', action='store_true', default=False,
                      help='是否使用模式识别 (默认: True)')
    parser.add_argument('--train_patterns', action='store_true', default=False,
                      help='是否为不同模式单独训练模型 (默认: False)')

    # parser.add_argument('--data_file', type=str, default='加密_2024年四省一市负荷预测数据.xlsx',
    #                   help='原始负荷数据文件路径 (默认: None)')
    parser.add_argument('--province', type=str, default='上海',
                      help='要处理的省份名称 (默认: 上海)')
    parser.add_argument('--generate_ts', action='store_true', default=False,
                      help='是否生成时间序列数据 (默认: False)')
    
    # 添加高峰感知相关参数
    parser.add_argument('--peak_aware', action='store_true', default=True,
                      help='是否使用高峰低谷感知功能 (默认: False)')
    parser.add_argument('--peak_start', type=int, default=7,
                      help='高峰时段开始小时 (默认: 8)')
    parser.add_argument('--peak_end', type=int, default=22,
                      help='高峰时段结束小时 (默认: 22)')
    parser.add_argument('--peak_weight', type=float, default=10,
                      help='高峰时段损失权重 (默认: 2.5)')
    
    parser.add_argument('--valley_start', type=int, default=0,
                      help='低谷时段开始小时 (默认: 0)')
    parser.add_argument('--valley_end', type=int, default=6,
                      help='低谷时段结束小时 (默认: 7)')
    parser.add_argument('--valley_weight', type=float, default=1.5,
                      help='低谷损失权重 (默认: 1.5)')
    parser.add_argument('--analyze_peaks', action='store_true', default=True,
                      help='是否分析预测性能 (默认: True)')    
    
    # 添加其他参数
    parser.add_argument('--combine_forecast', action='store_true', default=False,
                    help='是否使用一般模型和高峰模型的结合 (默认: False)')
    parser.add_argument('--typical_days', action='store_true', default=False,
                    help='是否使用典型日数据 (默认: False)')
    parser.add_argument('--real_time_adjustment', action='store_true', default=False,
                    help='是否进行实时数据修正 (默认: False)')
    
    # 添加模型参数
    parser.add_argument('--model', type=str, default='convtrans',
                      help='模型类型 (默认: convtrans)')
    
    # 添加训练参数
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数 (默认: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值 (默认: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小 (默认: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
                        
    # 新增日前预测相关参数
    parser.add_argument('--day_ahead', action='store_true', default=False,
                      help='是否执行日前预测而非滚动预测 (默认: False)')
    parser.add_argument('--forecast_date', type=str, default=None,
                      help='日前预测的起始日期，格式YYYY-MM-DD (默认: 当前日期的下一天)')
    parser.add_argument('--forecast_end_date', type=str, default=None,
                      help='日前预测的结束日期，格式YYYY-MM-DD (默认: 与起始日期相同，即只预测一天)')
    
    # 增强平滑相关参数
    parser.add_argument('--enhanced_smoothing', action='store_true', default=False,
                      help='使用增强平滑算法进行日前预测 (默认: False)')
    parser.add_argument('--smoothing_window', type=int, default=24,
                      help='平滑窗口大小(点数) (默认: 24)')
    parser.add_argument('--historical_days', type=int, default=8,
                      help='用于模型输入的历史数据天数 (默认: 8)')
    parser.add_argument('--max_diff_pct', type=float, default=5.0,
                      help='允许的预测与实际值最大差异百分比 (默认: 5.0)')

    # --- 修改：从probabilistic改为train_prediction_type，并添加选项 ---
    parser.add_argument('--train_prediction_type', type=str, choices=['deterministic', 'probabilistic', 'interval'], 
                        default='deterministic',
                        help='训练的目标类型: deterministic=点预测, probabilistic=概率预测(分位数), interval=区间预测(误差分布) (默认: deterministic)')
    # ------------------------------------------------------------

    parser.add_argument('--quantiles', type=str, default='0.1,0.5,0.9',
                    help='概率预测的分位数，逗号分隔 (默认: 0.1,0.5,0.9)')
    
    # 区间预测相关参数
    # parser.add_argument('--interval_forecast', action='store_true', default=False,
    #                help='是否使用区间预测 (基于误差分布分析) (默认: False)')
    # --- 备注：interval_forecast 参数可以移除或保留，训练时使用 train_prediction_type 控制 --- 
    parser.add_argument('--confidence_level', type=float, default=0.9,
                   help='区间预测的置信水平 (默认: 0.9)')
    
    # 新增输出JSON参数
    parser.add_argument('--output_json', type=str, default=None,
                      help='将预测结果输出为JSON文件的路径')
    
    # 添加PID修正参数
    parser.add_argument('--enable_pid_correction', action='store_true', default=False,
                    help='是否启用PID误差修正 (默认: False)')
    parser.add_argument('--pretrain_days', type=int, default=3,
                    help='PID预训练天数 (默认: 3)')
    parser.add_argument('--window_size_hours', type=int, default=72,
                    help='误差分析滑动窗口大小(小时) (默认: 72)')
    parser.add_argument('--enable_adaptation', type=str_to_bool, nargs='?', const=True, default=True, help='是否启用PID参数自适应调整')

    # Add initial PID parameters from user
    parser.add_argument('--initial_kp', type=float, default=None, help='PID控制器的初始Kp值')
    parser.add_argument('--initial_ki', type=float, default=None, help='PID控制器的初始Ki值')
    parser.add_argument('--initial_kd', type=float, default=None, help='PID控制器的初始Kd值')

    # 添加天气感知预测相关参数
    parser.add_argument('--weather_aware', action='store_true', default=False,
                      help='是否使用天气感知预测 (默认: False)')
    parser.add_argument('--weather_data_path', type=str, default=None,
                      help='天气数据文件路径 (如果不指定，将根据省份自动构建)')
    parser.add_argument('--weather_features', type=str, default='temperature,humidity,pressure,wind_speed,wind_direction,precipitation,solar_radiation',
                      help='使用的天气特征，逗号分隔 (默认: temperature,humidity,pressure,wind_speed,wind_direction,precipitation,solar_radiation)')
    parser.add_argument('--weather_model_dir', type=str, default=None,
                      help='天气感知模型目录路径 (如果不指定，将根据省份自动构建)')

    # 添加可再生能源增强预测相关参数
    parser.add_argument('--renewable_enhanced', action='store_true', default=False,
                      help='是否使用可再生能源增强区间预测 (默认: False)')
    parser.add_argument('--enable_renewable_prediction', action='store_true', default=True,
                      help='是否启用新能源出力预测 (默认: True)')
    parser.add_argument('--renewable_types', type=str, default='pv,wind',
                      help='要预测的新能源类型，逗号分隔 (默认: pv,wind)')

    args = parser.parse_args()

    # Immediately flush critical startup messages
    print("--- DEBUG: scene_forecasting.py 脚本开始执行 ---", flush=True)
    print(f"--- DEBUG: Python 可执行文件路径: {sys.executable} ---", flush=True)
    print(f"--- DEBUG: sys.path: {sys.path} ---", flush=True)

    # -----------------------------导入自定义模块-------------------------------
    from scripts.forecast.interval_forecast_fixed import perform_interval_forecast, perform_interval_forecast_for_range
    
    # 导入天气感知预测函数
    if args.weather_aware and args.forecast_type in ['load', 'pv', 'wind']:
        from scripts.forecast.day_ahead_forecast import perform_weather_aware_day_ahead_forecast
        from scripts.forecast.forecast import perform_weather_aware_rolling_forecast
        from scripts.forecast.weather_aware_interval_forecast import perform_weather_aware_interval_forecast_for_range
        print(f"--- DEBUG: Successfully imported weather-aware forecast functions for {args.forecast_type} ---", flush=True)
    
    # 导入可再生能源增强预测函数
    if args.renewable_enhanced and args.train_prediction_type == 'interval':
        from scripts.forecast.enhanced_interval_forecast_with_renewables import perform_enhanced_interval_forecast_with_renewables
        print("--- DEBUG: Successfully imported renewable-enhanced interval forecast functions ---", flush=True)
    
    # 根据是否执行日前预测，选择导入不同的预测函数
    if args.day_ahead:
        from scripts.forecast.day_ahead_forecast import perform_day_ahead_forecast, perform_day_ahead_forecast_with_smooth_transition, perform_day_ahead_forecast_with_enhanced_smoothing
    elif args.train_prediction_type == 'probabilistic':
            # -------------------------------------------
        from scripts.forecast.probabilistic_forecast import perform_probabilistic_forecast

    elif args.train_prediction_type == 'interval':
        # from scripts.train.train_probabilistic import train_probabilistic_model
        # print("--- DEBUG: Skipped importing train functions ---", flush=True)
        # 导入简化版区间预测
        # from scripts.forecast.interval_forecast_simple import perform_interval_forecast
        
        # 导入区间预测工具
        from utils.interval_forecast_utils import DataPipeline, create_prediction_intervals, plot_interval_forecast
    else:
        from scripts.forecast.forecast import perform_rolling_forecast, perform_rolling_forecast_with_patterns, perform_rolling_forecast_with_peak_awareness, perform_combined_forecast
        from scripts.forecast.forecast import enhanced_combined_forecast, perform_rolling_forecast_with_non_peak_awareness, perform_rolling_forecast_with_peak_awareness_and_real_time_adjustment
        from scripts.forecast.error_correction_forecast import advanced_error_correction_system
        # from scripts.train.train_torch import train_forecast_model, train_forecast_model_with_peak_awareness, train_pattern_specific_models, train_forecast_model_with_non_peak_awareness

    if args.model == 'keras':
        # 只有在使用Keras模型时才导入
        if args.mode == 'train':
            from scripts.train.train_keras import train_keras_model
    else:
        # 使用PyTorch模型时导入
        if args.mode == 'train':
            from scripts.train.train_torch import train_forecast_model, train_forecast_model_with_peak_awareness
        from models.torch_models import TorchConvTransformer, PeakAwareConvTransformer, ProbabilisticConvTransformer, IntervalPeakAwareConvTransformer

    from utils.evaluator import plot_peak_forecast_analysis, calculate_metrics

    # -----------------------------设置参数-------------------------------

    # --- Construct data_path based on type and province if not provided ---
    if args.data_path is None:
        if not args.province:
            # Critical error, should be flushed if it exits
            print("错误: 必须提供 --province 参数才能自动构建数据路径", flush=True)
            raise ValueError("必须提供 --province 参数才能自动构建数据路径")
        temp_dataset_id = args.province
        args.data_path = f"data/timeseries_{args.forecast_type}_{temp_dataset_id}.csv"
        print(f"自动构建数据路径: {args.data_path}", flush=True)

    # 获取数据集ID (现在基于省份参数) - 提前设置，避免后续使用时出错
    if args.province:
         args.dataset_id = args.province # Use province directly as dataset_id
    else:
         # Try to extract from data_path if province wasn't given but data_path was
         data_filename = os.path.basename(args.data_path)
         if f'timeseries_{args.forecast_type}_' in data_filename:
             args.dataset_id = data_filename.replace(f'timeseries_{args.forecast_type}_', '').replace('.csv', '')
         else:
             # Fallback or raise error if ID cannot be determined
             raise ValueError("无法确定数据集ID (省份)")

    # 构建天气数据路径（如果启用天气感知且未指定路径）
    if args.weather_aware and args.weather_data_path is None:
        if not args.province:
            print("错误: 启用天气感知预测时必须提供 --province 参数才能自动构建天气数据路径", flush=True)
            raise ValueError("启用天气感知预测时必须提供 --province 参数才能自动构建天气数据路径")
        # 构建天气数据路径，根据预测类型使用对应的天气数据文件
        args.weather_data_path = f"data/timeseries_{args.forecast_type}_weather_{args.province}.csv"
        print(f"自动构建天气数据路径: {args.weather_data_path}", flush=True)
        
        # 检查天气数据文件是否存在
        if not os.path.exists(args.weather_data_path):
            print(f"警告: 天气数据文件 {args.weather_data_path} 不存在", flush=True)
            print("请确保天气数据文件存在，或手动指定 --weather_data_path 参数", flush=True)
    
    # 构建天气模型目录路径（如果启用天气感知且未指定路径）
    if args.weather_aware and args.weather_model_dir is None:
        # 统一使用 convtrans_weather 作为天气感知模型的目录名
        args.weather_model_dir = f"models/convtrans_weather/{args.forecast_type}/{args.dataset_id}"
        print(f"自动构建天气模型目录: {args.weather_model_dir}", flush=True)

    # 打印关键参数信息，帮助调试
    print("\n=====================================================", flush=True)
    print("关键参数信息:", flush=True)
    print(f"预测类型: {args.forecast_type}", flush=True)
    print(f"数据路径: {args.data_path}", flush=True)
    print(f"数据集ID (省份): {args.dataset_id}", flush=True)
    print(f"操作模式: {args.mode}", flush=True)
    print(f"日前预测: {args.day_ahead}", flush=True)
    print(f"预测日期: {args.forecast_date}", flush=True)
    print(f"输出JSON: {args.output_json}", flush=True)
    print(f"增强平滑: {args.enhanced_smoothing}", flush=True)
    # --- 修改：使用新的参数判断是否为概率预测 --- 
    print(f"概率预测模式: {(args.train_prediction_type == 'probabilistic')}", flush=True)
    if args.train_prediction_type == 'probabilistic':
    # -------------------------------------------
         print(f"  分位数: {args.quantiles}", flush=True)
    # --- 修改：添加区间预测模式的打印 --- 
    print(f"区间预测模式: {(args.train_prediction_type == 'interval')}", flush=True)
    if args.train_prediction_type == 'interval':
        print(f"  置信水平: {args.confidence_level}", flush=True)
    print(f"启用PID修正: {args.enable_pid_correction}", flush=True)
    if args.enable_pid_correction:
        print(f"  PID预训练天数: {args.pretrain_days}", flush=True)
        print(f"  PID窗口大小(小时): {args.window_size_hours}", flush=True)
        print(f"  PID启用自适应: {args.enable_adaptation}", flush=True)
        if args.initial_kp is not None:
            print(f"  用户提供的初始PID: Kp={args.initial_kp}, Ki={args.initial_ki}, Kd={args.initial_kd}", flush=True)
    print(f"天气感知预测: {args.weather_aware}", flush=True)
    if args.weather_aware:
        print(f"  天气数据路径: {args.weather_data_path}", flush=True)
        print(f"  天气特征: {args.weather_features}", flush=True)
        print(f"  天气模型目录: {args.weather_model_dir}", flush=True)
    print(f"可再生能源增强预测: {args.renewable_enhanced}", flush=True)
    if args.renewable_enhanced:
        print(f"  启用新能源预测: {args.enable_renewable_prediction}", flush=True)
        print(f"  新能源类型: {args.renewable_types}", flush=True)
    # ------------------------------------
    print("=====================================================\n", flush=True)
    
    # 记录开始时间
    start_time = time.time()


    # 创建必要目录，将 forecast_type 加入路径
    # --- 修改：根据训练类型决定模型名称 --- 
    if args.train_prediction_type == 'probabilistic':
         model_base_name = ProbabilisticConvTransformer.model_type
    elif args.train_prediction_type == 'interval':
         # 假设区间模型也使用峰谷感知的 Transformer 结构
         model_base_name = IntervalPeakAwareConvTransformer.model_type # 使用区间模型类名
    elif args.peak_aware:
         model_base_name = PeakAwareConvTransformer.model_type
    else:
         model_base_name = TorchConvTransformer.model_type
    # ----------------------------------

    base_model_dir = f"models/{model_base_name}/{args.forecast_type}/{args.dataset_id}"
    base_scaler_dir = f"models/scalers/{model_base_name}/{args.forecast_type}/{args.dataset_id}"
    base_results_dir = f"results/{model_base_name}/{args.forecast_type}/{args.dataset_id}"
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_scaler_dir, exist_ok=True)
    os.makedirs(base_results_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True) # data 目录不需要区分类型

    # 如果使用天气感知功能，创建相应目录
    if args.weather_aware:
        weather_model_dir = f"models/convtrans_weather/{args.forecast_type}/{args.dataset_id}"
        weather_scaler_dir = f"models/scalers/convtrans_weather/{args.forecast_type}/{args.dataset_id}"
        weather_results_dir = f"results/convtrans_weather/{args.forecast_type}/{args.dataset_id}"
        os.makedirs(weather_model_dir, exist_ok=True)
        os.makedirs(weather_scaler_dir, exist_ok=True)
        os.makedirs(weather_results_dir, exist_ok=True)

    # 如果使用高峰感知功能，创建相应目录
    if args.peak_aware:
        peak_model_dir = f"models/convtrans_peak/{args.forecast_type}/{args.dataset_id}"
        peak_scaler_dir = f"models/scalers/convtrans_peak/{args.forecast_type}/{args.dataset_id}"
        peak_results_dir = f"results/convtrans_peak/{args.forecast_type}/{args.dataset_id}"
        os.makedirs(peak_model_dir, exist_ok=True)
        os.makedirs(peak_scaler_dir, exist_ok=True)
        os.makedirs(peak_results_dir, exist_ok=True)

    # 为日前预测结果创建目录 (路径中也加入 forecast_type)
    day_ahead_results_dir = f"results/day_ahead/{args.forecast_type}/{args.dataset_id}"
    prob_results_dir = f"results/probabilistic/{args.forecast_type}/{args.dataset_id}"
    os.makedirs(day_ahead_results_dir, exist_ok=True)
    os.makedirs(prob_results_dir, exist_ok=True)

    # 导入时间序列数据
    data_path = args.data_path
    if not os.path.exists(data_path):
        # Critical error, should be flushed
        print(f"错误: 数据文件 {data_path} 不存在，请检查路径", flush=True)
        raise FileNotFoundError(f"数据文件 {data_path} 不存在，请检查路径")

    convtrans_config = {
        'seq_length': int(1440 / args.interval),  # 24小时数据，默认15分钟即96点
        'pred_length': 1,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'patience': args.patience,
        'valley_weight': args.valley_weight,
        'peak_weight': args.peak_weight,
        'valley_hours': (args.valley_start, args.valley_end),
        'peak_hours': (args.peak_start, args.peak_end),
        'use_peak_loss': True
    }

    # 根据模式执行相应操作
    if args.mode in ['train', 'both']:
        print(f"=== 开始 {args.forecast_type} 模型训练 ===", flush=True)

        try:
            from scripts.train.train_torch import train_forecast_model, train_forecast_model_with_peak_awareness
            # --- 确认导入概率训练函数 ---
            from scripts.train.train_probabilistic import train_probabilistic_model
            # --- 天气感知训练函数导入 ---
            if args.weather_aware:
                from scripts.train.train_torch import train_weather_aware_model
                print("--- DEBUG: Successfully imported weather-aware train functions ---", flush=True)
            # --- 确认/添加区间训练函数导入 (如果存在) ---
            # from scripts.train.train_interval import train_interval_model # 假设存在此文件和函数
            # -----------------------------
            print("--- DEBUG: Successfully imported train functions ---", flush=True)
        except ImportError as e:
            print(f"错误：无法导入训练函数: {e}", flush=True)
            print("请确保 train_torch.py 和 train_probabilistic.py 文件存在于 scripts/train 目录下。", flush=True)
            if args.weather_aware:
                print("请确保 train_weather_aware.py 文件存在于 scripts/train 目录下。", flush=True)
            sys.exit(1)
        
        # --- Choose Training Function Based on Flags --- 
        # --- 天气感知训练逻辑 ---
        if args.weather_aware and args.forecast_type in ['load', 'pv', 'wind']:
            print(f"训练模式: 天气感知{args.forecast_type}预测模型", flush=True)
            print(f"天气数据路径: {args.weather_data_path}", flush=True)
            print(f"天气特征: {args.weather_features}", flush=True)
            print(f"模型保存目录: {args.weather_model_dir}", flush=True)
            
            # 解析天气特征
            weather_features = [f.strip() for f in args.weather_features.split(',')]
            
            if args.forecast_type == 'load':
                # 负荷预测使用原有的训练函数
                train_weather_aware_model(
                    data_path=args.weather_data_path,  # 使用天气数据路径
                    train_start_date=args.train_start,
                    train_end_date=args.train_end,
                    forecast_type=args.forecast_type,
                    weather_features=weather_features,
                    convtrans_config=convtrans_config,
                    retrain=args.retrain,
                    dataset_id=args.dataset_id,
                    peak_hours=convtrans_config['peak_hours'],
                    valley_hours=convtrans_config['valley_hours'],
                    peak_weight=convtrans_config['peak_weight'],
                    valley_weight=convtrans_config['valley_weight']
                )
            else:
                # 光伏和风电使用新的训练函数（现在与负荷预测保持一致的参数结构）
                from scripts.train.train_weather_aware_renewable import train_weather_aware_renewable_model
                train_weather_aware_renewable_model(
                    data_path=args.weather_data_path,
                    train_start_date=args.train_start,
                    train_end_date=args.train_end,
                    forecast_type=args.forecast_type,
                    weather_features=weather_features,
                    convtrans_config=convtrans_config,
                    retrain=args.retrain,
                    dataset_id=args.dataset_id,
                    peak_hours=convtrans_config['peak_hours'],
                    valley_hours=convtrans_config['valley_hours'],
                    peak_weight=convtrans_config['peak_weight'],
                    valley_weight=convtrans_config['valley_weight']
                )
        # --- 修改：根据 train_prediction_type 调用不同函数 ---
        elif args.train_prediction_type == 'probabilistic':
            print("训练模式: 概率模型 (Quantile Regression)", flush=True)
            # Parse quantiles
            try:
                quantiles = [float(q.strip()) for q in args.quantiles.split(',')] 
                if not quantiles or any(q <= 0 or q >= 1 for q in quantiles): 
                    raise ValueError("分位数必须在 (0, 1) 之间")
            except ValueError as e:
                print(f"错误: 无效的分位数 '{args.quantiles}': {e}", flush=True)
                sys.exit(1) # Exit if quantiles are invalid
                
            train_probabilistic_model(
                data_path=args.data_path,
                train_start_date=args.train_start,
                train_end_date=args.train_end,
                convtrans_config=convtrans_config,
                retrain=args.retrain,
                dataset_id=args.dataset_id,
                forecast_type=args.forecast_type,
                quantiles=quantiles
            )
        elif args.train_prediction_type == 'interval':
            print("训练模式: 区间预测模型", flush=True)
            # 检查是否为负荷预测且启用峰谷感知，因为当前区间模型基于此
            if args.forecast_type == 'load' and args.peak_aware:
                print(" - 使用峰谷感知训练函数训练区间模型", flush=True)
                # !! 注意：这里复用了峰谷感知训练函数。需要确保此函数能正确处理和保存 IntervalPeakAwareConvTransformer !!
                train_forecast_model_with_peak_awareness(
                     data_path=args.data_path,
                     train_start_date=args.train_start,
                     train_end_date=args.train_end,
                     convtrans_config=convtrans_config,
                     retrain=args.retrain,
                     dataset_id=args.dataset_id,
                     forecast_type=args.forecast_type,
                     peak_hours=convtrans_config['peak_hours'],
                     valley_hours=convtrans_config['valley_hours'],
                     peak_weight=convtrans_config['peak_weight'],
                     valley_weight=convtrans_config['valley_weight'],
                     # 可能需要传递一个参数指定要实例化的模型类
                     # model_class=IntervalPeakAwareConvTransformer # 类似这样，但需要函数支持
                )
            else:
                print("错误：当前区间预测模型实现依赖于负荷数据的峰谷感知特性。请启用 --peak_aware 并设置 --forecast_type load。", flush=True)
                sys.exit(1)
            # 如果有专门的区间训练函数，应该在这里调用：
            # train_interval_model(...)
        elif args.train_prediction_type == 'deterministic':
            if args.peak_aware:
                print("训练模式: 确定性高峰感知模型", flush=True)
                train_forecast_model_with_peak_awareness(
                    data_path=args.data_path,
                    train_start_date=args.train_start,
                    train_end_date=args.train_end,
                    convtrans_config=convtrans_config,
                    retrain=args.retrain,
                    dataset_id=args.dataset_id,
                    forecast_type=args.forecast_type,
                    peak_hours=convtrans_config['peak_hours'],
                    valley_hours=convtrans_config['valley_hours'],
                    peak_weight=convtrans_config['peak_weight'],
                    valley_weight=convtrans_config['valley_weight']
                )
            else:
                print(f"训练模式: 确定性 {args.forecast_type} 通用模型", flush=True)
                train_forecast_model(
                    data_path=args.data_path,
                    train_start_date=args.train_start,
                    train_end_date=args.train_end,
                    convtrans_config=convtrans_config,
                    retrain=args.retrain,
                    dataset_id=args.dataset_id,
                    forecast_type=args.forecast_type
                )
        else:
            print(f"错误: 未知的训练目标类型 '{args.train_prediction_type}'", flush=True)
            sys.exit(1)
        # -----------------------------------------------------
        # --------------------------------------------

    results_df = None
    
    if args.mode in ['forecast', 'both']:
        # 打印预测日期信息进行确认
        print("\n===========================", flush=True)
        print("预测日期信息确认:", flush=True)
        if args.day_ahead:
            print(f"预测类型: {args.forecast_type}", flush=True)
            # 如果未指定预测日期，默认为明天 (或基于训练结束日期)
            if args.forecast_date is None:
                # Try to base off train_end_date if available, otherwise today+1
                try:
                     base_date = datetime.strptime(args.train_end, '%Y-%m-%d')
                except:
                     base_date = datetime.now()
                forecast_date = (base_date + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                forecast_date = args.forecast_date
            
            # 处理结束日期参数
            if args.forecast_end_date is None:
                forecast_end_date = forecast_date  # 默认与开始日期相同
                print(f"预测日期: {forecast_date}", flush=True)
            else:
                forecast_end_date = args.forecast_end_date
                print(f"预测日期范围: {forecast_date} 至 {forecast_end_date}", flush=True)
            
            print(f"历史数据天数: {args.historical_days}", flush=True)
        else:
            print(f"预测类型: 滚动预测", flush=True)
            print(f"预测开始日期: {args.forecast_start}", flush=True)
            print(f"预测结束日期: {args.forecast_end}", flush=True)
            print(f"历史数据天数: {args.historical_days}", flush=True)
            forecast_date = None # Not used for rolling
            forecast_end_date = None # Not used for rolling
        
        # --- Determine Forecast Method --- 
        # --- 修改：使用 train_prediction_type 判断预测类型 ---
        if args.train_prediction_type == 'probabilistic' and args.day_ahead:
        # --------------------------------------------------
             forecast_method_str = "日前概率预测"
             print(f"预测方法: {forecast_method_str}", flush=True)
             print("===========================\n", flush=True)
             # Perform probabilistic forecast
             print(f"=== 开始 {args.forecast_type} 日前概率预测 ===", flush=True)
             try:
                 # Parse quantiles again for prediction
                 quantiles = [float(q.strip()) for q in args.quantiles.split(',')] 
                 results_df = perform_probabilistic_forecast(
                     data_path=args.data_path,
                     forecast_date=forecast_date,
                     dataset_id=args.dataset_id,
                     forecast_type=args.forecast_type,
                     quantiles=quantiles,
                     interval_minutes=args.interval,
                     peak_hours=convtrans_config['peak_hours'], # Still needed for features
                     valley_hours=convtrans_config['valley_hours'],
                     historical_days=args.historical_days  # 添加历史天数参数
                 )
             except Exception as e:
                 print(f"{args.forecast_type} 日前概率预测失败: {e}", flush=True)
                 import traceback
                 traceback.print_exc(file=sys.stdout) # Ensure traceback goes to capturable stdout
                 print("请确保对应的概率模型已训练并可用。", flush=True)
        # --- 修改：使用 train_prediction_type 判断 ---
        elif args.train_prediction_type == 'probabilistic' and not args.day_ahead:
        # -----------------------------------------
             print("错误: 当前不支持滚动概率预测。请使用 --day_ahead 进行概率预测。", flush=True)
             sys.exit(1)
        # --- 修改：使用 train_prediction_type 判断 ---
        elif args.train_prediction_type == 'interval' and args.day_ahead:
        # -----------------------------------------
             if args.renewable_enhanced:
                 forecast_method_str = "日前可再生能源增强区间预测"
                 print(f"预测方法: {forecast_method_str}", flush=True)
                 print("===========================\n", flush=True)
                 print(f"=== 开始 {args.forecast_type} 日前可再生能源增强区间预测 ===", flush=True)
                 try:
                    start_date_str = args.forecast_date if args.forecast_date else forecast_date
                    end_date_str = args.forecast_end_date if args.forecast_end_date else forecast_end_date
                    
                    # 调用增强的区间预测函数
                    results_df, interval_metrics = perform_enhanced_interval_forecast_with_renewables(
                        province=args.dataset_id,
                        forecast_date=start_date_str,
                        forecast_end_date=end_date_str,
                        confidence_level=args.confidence_level,
                        historical_days=args.historical_days,
                        enable_renewable_prediction=args.enable_renewable_prediction
                    )
                    metrics = interval_metrics
                    
                    # 确保enhanced_scenario信息在metrics中被正确处理
                    if 'enhanced_scenario' in interval_metrics:
                        print(f"增强场景信息已包含在interval_metrics中", flush=True)
                        scenario_info = interval_metrics['enhanced_scenario']
                        if 'enhanced_scenario' in scenario_info:
                            print(f"识别增强场景: {scenario_info['enhanced_scenario']['name']}", flush=True)
                        if 'composite_risk_level' in scenario_info:
                            print(f"综合风险等级: {scenario_info['composite_risk_level']}", flush=True)
                    else:
                        print("警告: interval_metrics中未找到enhanced_scenario信息", flush=True)
                    
                    # 确保renewable_predictions信息被包含
                    if 'renewable_predictions' in interval_metrics:
                        renewable_data = interval_metrics['renewable_predictions']
                        if renewable_data and isinstance(renewable_data, dict):
                            pv_status = renewable_data.get('pv', {}).get('error', '成功')
                            wind_status = renewable_data.get('wind', {}).get('error', '成功')
                            print(f"新能源预测状态: PV={pv_status}, Wind={wind_status}", flush=True)
                    
                    print(f"可再生能源增强区间预测完成", flush=True)
                 except Exception as e:
                     print(f"{args.forecast_type} 可再生能源增强区间预测失败: {e}", flush=True)
                     import traceback
                     traceback.print_exc(file=sys.stdout)
                     print("请确保对应的天气感知模型和新能源模型已训练并可用。", flush=True)
                     
             elif args.weather_aware and args.forecast_type in ['load', 'pv', 'wind']:
                 forecast_method_str = "日前天气感知区间预测"
                 print(f"预测方法: {forecast_method_str}", flush=True)
                 print("===========================\n", flush=True)
                 print(f"=== 开始 {args.forecast_type} 日前天气感知区间预测 ===", flush=True)
                 try:
                    start_date_str = args.forecast_date if args.forecast_date else forecast_date
                    end_date_str = args.forecast_end_date if args.forecast_end_date else forecast_end_date
                    
                    results_df, interval_metrics = perform_weather_aware_interval_forecast_for_range(
                        province=args.dataset_id,
                        forecast_type=args.forecast_type,
                        start_date_str=start_date_str,
                        end_date_str=end_date_str,
                        confidence_level=args.confidence_level,
                        historical_days=args.historical_days
                    )
                    metrics = interval_metrics
                    
                    # 确保weather_scenario信息在metrics中被正确处理
                    if 'weather_scenario' in interval_metrics:
                        print(f"天气感知场景信息已包含在interval_metrics中", flush=True)
                    else:
                        print("警告: interval_metrics中未找到weather_scenario信息", flush=True)
                    print(f"天气感知区间预测完成", flush=True)
                 except Exception as e:
                     print(f"{args.forecast_type} 天气感知区间预测失败: {e}", flush=True)
                     import traceback
                     traceback.print_exc(file=sys.stdout)
                     print("请确保对应的天气感知模型已训练并可用。", flush=True)

             else:
                forecast_method_str = "日前区间预测"
                print(f"预测方法: {forecast_method_str}", flush=True)
                print("===========================\n", flush=True)
                # 执行区间预测
                print(f"=== 开始 {args.forecast_type} 日前区间预测 ===", flush=True)
                try:
                    # 解析日期范围
                    start_date_str = args.forecast_date if args.forecast_date else forecast_date
                    end_date_str = args.forecast_end_date if args.forecast_end_date else forecast_end_date

                    # 调用新的范围预测函数
                    results_df, interval_metrics = perform_interval_forecast_for_range(
                        data_path=args.data_path,
                        forecast_type=args.forecast_type,
                        province=args.dataset_id,
                        start_date_str=start_date_str, # 传递字符串形式的日期
                        end_date_str=end_date_str,   # 传递字符串形式的日期
                        n_intervals=args.historical_days,
                        model_path=base_model_dir, # 使用正确的模型基础目录
                        scaler_path=base_scaler_dir, # 使用正确的缩放器基础目录
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        quantiles=[(1-args.confidence_level)/2, 1-(1-args.confidence_level)/2],
                        rolling=False, # 日前预测通常不是滚动的
                        seq_length=convtrans_config['seq_length'],
                        peak_hours=convtrans_config['peak_hours'],
                        valley_hours=convtrans_config['valley_hours'],
                        fix_nan=True
                    )
                    
                    # 保存metrics到全局变量，以便后续在JSON输出中使用
                    metrics = interval_metrics
                    
                    # 确保weather_scenario信息在metrics中被正确处理
                    if 'weather_scenario' in interval_metrics:
                        print(f"场景信息已包含在interval_metrics中: {type(interval_metrics['weather_scenario'])}", flush=True)
                    else:
                        print("警告: interval_metrics中未找到weather_scenario信息", flush=True)
                    
                    print(f"区间预测完成", flush=True)
                    
                except Exception as e:
                    print(f"{args.forecast_type} 区间预测失败: {e}", flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stdout)
                    print("请确保对应的峰值感知模型已训练并可用。", flush=True)
        # --- 修改：使用 train_prediction_type 判断 ---
        elif args.train_prediction_type == 'interval' and not args.day_ahead:
        # -----------------------------------------
             print("错误: 当前不支持滚动区间预测。请使用 --day_ahead 进行区间预测。", flush=True)
             sys.exit(1)
        # --- 修改：仅处理确定性预测和日前的组合 --- 
        elif args.train_prediction_type == 'deterministic' and args.day_ahead:
        # ------------------------------------------
             forecast_method_str = "日前点预测"
             if args.enhanced_smoothing:
                 forecast_method_str += " (增强平滑)"
             else:
                 forecast_method_str += " (基础平滑)"
             print(f"预测方法: {forecast_method_str}", flush=True)
             print("===========================\n", flush=True)
             print(f"=== 开始 {args.forecast_type} 日前点预测 ===", flush=True)
             try:
                 # 添加调试信息
                 print("\n日前预测配置信息:", flush=True)
                 print(f"  - 默认seq_length: {convtrans_config['seq_length']}", flush=True)
                 print(f"  - 时间间隔: {args.interval}分钟", flush=True)
                 print(f"  - 一天总点数: {int(24*60/args.interval)}", flush=True)
                 
                 # --- 天气感知预测逻辑 ---
                 if args.weather_aware and args.forecast_type in ['load', 'pv', 'wind']:
                     print("使用天气感知日前预测", flush=True)
                     print(f"天气数据路径: {args.weather_data_path}", flush=True)
                     print(f"天气特征: {args.weather_features}", flush=True)
                     
                     # 解析天气特征
                     weather_features = [f.strip() for f in args.weather_features.split(',')]
                     
                     # 处理日期范围
                     start_date_str = args.forecast_date if args.forecast_date else forecast_date
                     end_date_str = args.forecast_end_date if args.forecast_end_date else forecast_end_date
                     
                     weather_result = perform_weather_aware_day_ahead_forecast(
                         data_path=args.weather_data_path,  # 使用天气数据路径作为data_path
                         forecast_date=start_date_str,
                         forecast_end_date=end_date_str,
                         weather_features=weather_features,
                         peak_hours=convtrans_config['peak_hours'],
                         valley_hours=convtrans_config['valley_hours'],
                         peak_weight=convtrans_config['peak_weight'],
                         valley_weight=convtrans_config['valley_weight'],
                         dataset_id=args.dataset_id,
                         forecast_type=args.forecast_type,
                         historical_days=args.historical_days
                     )
                     
                     # 转换天气感知预测结果为DataFrame格式
                     if isinstance(weather_result, dict) and 'timestamps' in weather_result and 'predictions' in weather_result:
                         results_df = pd.DataFrame({
                             'datetime': weather_result['timestamps'],
                             'predicted': weather_result['predictions'],
                             'actual': weather_result.get('actuals', [None] * len(weather_result['predictions']))
                         })
                         
                         # 添加高峰时段标记
                         results_df['is_peak'] = results_df['datetime'].apply(
                             lambda x: convtrans_config['peak_hours'][0] <= pd.Timestamp(x).hour <= convtrans_config['peak_hours'][1]
                         ).astype(int)
                         
                         # 添加预测日期标识
                         results_df['prediction_date'] = start_date_str
                         
                         # 计算误差百分比（如果有实际值）
                         if 'actual' in results_df.columns and not results_df['actual'].isna().all():
                             results_df['error_pct'] = ((results_df['predicted'] - results_df['actual']) / results_df['actual'] * 100).round(2)
                         else:
                             results_df['error_pct'] = None
                             
                         # 保存天气感知预测的指标
                         if isinstance(weather_result, dict):
                             metrics = {
                                 'mae': weather_result.get('mae', 0),
                                 'rmse': weather_result.get('rmse', 0),
                                 'mape': weather_result.get('mape', 0)
                             }
                         
                         print(f"天气感知预测结果转换完成，数据形状: {results_df.shape}", flush=True)
                     else:
                         print("警告: 天气感知预测返回格式不正确，无法转换为DataFrame", flush=True)
                         results_df = None
                 else:
                     # 选择合适的日前预测方法
                     if args.enhanced_smoothing:
                         # 使用增强平滑算法
                         day_ahead_func = perform_day_ahead_forecast_with_enhanced_smoothing
                         extra_args = {
                             'smoothing_window': args.smoothing_window,
                             'use_historical_patterns': True
                         }
                     else:
                         # 使用基本平滑算法
                         day_ahead_func = perform_day_ahead_forecast_with_smooth_transition
                         extra_args = {
                         }
                         
                     results_df = day_ahead_func(
                         data_path=data_path,
                         forecast_date=forecast_date,
                         peak_hours=convtrans_config['peak_hours'],
                         valley_hours=convtrans_config['valley_hours'],
                         peak_weight=convtrans_config['peak_weight'],
                         valley_weight=convtrans_config['valley_weight'],
                         dataset_id=args.dataset_id,
                         max_allowed_diff_pct=args.max_diff_pct,
                         forecast_type=args.forecast_type,
                         forecast_end_date=forecast_end_date,  # 添加结束日期参数
                         historical_days=args.historical_days, # Ensure historical_days is passed here
                         **extra_args
                     )
             except Exception as e:
                 print(f"{args.forecast_type} 日前预测失败: {e}", flush=True)
                 import traceback
                 traceback.print_exc(file=sys.stdout)
                 print("请确保对应类型的模型已训练并可用", flush=True)
        # --- 修改：仅处理确定性预测和滚动的组合 --- 
        elif args.train_prediction_type == 'deterministic' and not args.day_ahead:
        # -------------------------------------------
            print(f"=== 开始 {args.forecast_type} 滚动预测 ===", flush=True)
            try:
                pid_params_dict = None
                if False: # Was if args.pid_params:
                    try:
                        pid_params_dict = json.loads(args.pid_params)
                        print(f"解析得到的PID参数: {pid_params_dict}", flush=True)
                    except json.JSONDecodeError as json_err:
                        print(f"错误: 解析PID参数JSON字符串失败: {json_err}. 将不使用特定PID参数。", flush=True)
                        pid_params_dict = None # Fallback

                # --- 天气感知滚动预测逻辑 ---
                if args.weather_aware and args.forecast_type in ['load', 'pv', 'wind']:
                    print("滚动预测方法: 天气感知", flush=True)
                    print(f"天气数据路径: {args.weather_data_path}", flush=True)
                    print(f"天气特征: {args.weather_features}", flush=True)
                    print(f"天气模型目录: {args.weather_model_dir}", flush=True)
                    
                    # 解析天气特征
                    weather_features = [f.strip() for f in args.weather_features.split(',')]
                    
                    results_df = perform_weather_aware_rolling_forecast(
                        data_path=args.weather_data_path,
                        start_date=args.forecast_start,
                        end_date=args.forecast_end,
                        forecast_interval=args.interval,
                        peak_hours=convtrans_config['peak_hours'],
                        valley_hours=convtrans_config['valley_hours'],
                        peak_weight=convtrans_config['peak_weight'],
                        valley_weight=convtrans_config['valley_weight'],
                        apply_smoothing=False,
                        dataset_id=args.dataset_id,
                        forecast_type=args.forecast_type,
                        historical_days=args.historical_days,
                        weather_features=weather_features
                    )
                # --- Start of the block to be replaced ---
                elif args.enable_pid_correction and args.forecast_type == 'load': # PID修正仅用于负荷预测
                    print("滚动预测方法: 高级误差修正 (PID)", flush=True)
                    rolling_func = advanced_error_correction_system
                    forecast_params = {
                        'data_path': data_path, 'start_date': args.forecast_start, 'end_date': args.forecast_end,
                        'forecast_interval': args.interval, 'dataset_id': args.dataset_id,
                        'forecast_type': args.forecast_type, 'peak_hours': convtrans_config['peak_hours'],
                        'valley_hours': convtrans_config['valley_hours'], 'peak_weight': convtrans_config['peak_weight'],
                        'valley_weight': convtrans_config['valley_weight'], 'apply_smoothing': False, 
                        'historical_days': args.historical_days,
                        # PID specific params from args
                        'pretrain_days': args.pretrain_days,
                        'window_size_hours': args.window_size_hours,
                        'enable_pid': True, # Explicitly enabling PID within this path
                        'enable_adaptation': args.enable_adaptation,
                        'enable_pattern_learning': False, # Default for now, can be parameterized
                        'enable_ensemble': False, # Default for now
                        # Pass initial PID parameters if provided
                        'initial_kp': args.initial_kp,
                        'initial_ki': args.initial_ki,
                        'initial_kd': args.initial_kd
                    }
                    results_df = rolling_func(**forecast_params)
                elif args.real_time_adjustment and args.forecast_type == 'load': # 简单实时校正仅用于负荷
                    print("滚动预测方法: 简单实时校正 (通过峰谷感知+实时调整实现)", flush=True)
                    rolling_func = perform_rolling_forecast_with_peak_awareness_and_real_time_adjustment
                    forecast_params = {
                        'data_path': data_path, 'start_date': args.forecast_start, 'end_date': args.forecast_end,
                        'forecast_interval': args.interval, 'dataset_id': args.dataset_id,
                        'peak_hours': convtrans_config['peak_hours'], 'valley_hours': convtrans_config['valley_hours'],
                        'peak_weight': convtrans_config['peak_weight'], 'valley_weight': convtrans_config['valley_weight'],
                        'apply_smoothing': False, 
                        'historical_days': args.historical_days,
                        'forecast_type': args.forecast_type
                    }
                    results_df = rolling_func(**forecast_params)
                elif args.peak_aware:
                    print("滚动预测方法: 峰谷感知", flush=True)
                    rolling_func = perform_rolling_forecast_with_peak_awareness
                    forecast_params = {
                        'data_path': data_path, 'start_date': args.forecast_start, 'end_date': args.forecast_end,
                        'forecast_interval': args.interval, 'dataset_id': args.dataset_id,
                        'peak_hours': convtrans_config['peak_hours'], 'valley_hours': convtrans_config['valley_hours'],
                        'peak_weight': convtrans_config['peak_weight'], 'valley_weight': convtrans_config['valley_weight'],
                        'apply_smoothing': False, 
                        'historical_days': args.historical_days,
                        'forecast_type': args.forecast_type
                    }
                    results_df = rolling_func(**forecast_params)
                elif args.use_patterns:
                    print("滚动预测方法: 模式识别", flush=True)
                    rolling_func = perform_rolling_forecast_with_patterns
                    forecast_params = {
                        'data_path': data_path, 'start_date': args.forecast_start, 'end_date': args.forecast_end,
                        'forecast_interval': args.interval, 'dataset_id': args.dataset_id,
                        'historical_days': args.historical_days, 'forecast_type': args.forecast_type
                    }
                    results_df = rolling_func(**forecast_params)
                else:
                    print("滚动预测方法: 标准滚动", flush=True)
                    rolling_func = perform_rolling_forecast
                    forecast_params = {
                        'data_path': data_path, 'start_date': args.forecast_start, 'end_date': args.forecast_end,
                        'forecast_interval': args.interval, 'apply_smoothing': False, 
                        'dataset_id': args.dataset_id, 'historical_days': args.historical_days,
                        'forecast_type': args.forecast_type
                    }
                    results_df = rolling_func(**forecast_params)
                # --- End of new correct logic ---
            except FileNotFoundError as e:
                print(f"错误: {e}", flush=True); print(f"请先训练 {args.forecast_type} 模型或提供已训练的模型文件", flush=True)
            except Exception as e:
                print(f"{args.forecast_type} 滚动预测失败: {e}", flush=True); import traceback; traceback.print_exc(file=sys.stdout)
                print("请确保对应类型的模型已训练并可用", flush=True)
        else:
            print(f"错误：当前不支持 {args.train_prediction_type} 类型的滚动预测。", flush=True); sys.exit(1)
        # ------------------------------
    
    # 计算执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n执行完成，用时 {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)", flush=True)

    # 初始化空的指标字典，避免后续引用前未赋值错误
    metrics = {}

    # 如果需要分析高峰时段预测性能
    if args.analyze_peaks and results_df is not None and isinstance(results_df, pd.DataFrame) and not results_df.empty and not (args.train_prediction_type == 'probabilistic'):
        print(f"=== 分析 {args.forecast_type} 点预测性能 ===", flush=True)
        
        # 增强数据有效性检查
        has_valid_data = (
            'actual' in results_df.columns and 
            'predicted' in results_df.columns and
            results_df['actual'].notna().sum() > 0 and 
            results_df['predicted'].notna().sum() > 0
        )
        
        if not has_valid_data:
            print("警告: 没有足够的有效数据进行预测性能分析，跳过此步骤", flush=True)
        else:
            # 确保包含 is_peak 列
            if 'is_peak' not in results_df.columns:
                results_df['is_peak'] = results_df['datetime'].apply(
                    lambda x: convtrans_config['peak_hours'][0] <= pd.Timestamp(x).hour <= convtrans_config['peak_hours'][1]
                ).astype(int)

            # 构建保存路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_results_dir = f"results/analysis/{args.forecast_type}/{args.dataset_id}"
            os.makedirs(analysis_results_dir, exist_ok=True)
            
            # 根据预测类型选择不同的分析标题和文件名
            if args.day_ahead:
                plot_title = f'{args.dataset_id} {args.forecast_type} 日前预测分析 ({forecast_date})'
                save_prefix = f'day_ahead_analysis_{forecast_date}'
            else:
                plot_title = f'{args.dataset_id} {args.forecast_type} 滚动预测分析 ({args.forecast_start} to {args.forecast_end})'
                save_prefix = f'rolling_analysis_{args.forecast_start}_{args.forecast_end}'
            
            try:
                # 执行分析并绘图
                plot_peak_forecast_analysis(
                    results_df,
                    title=plot_title,
                    save_path=f"{analysis_results_dir}/{save_prefix}_{timestamp}.png"
                )
                
                # 计算整体指标
                valid_data = results_df.dropna(subset=['actual', 'predicted'])
                if not valid_data.empty:
                    metrics = calculate_metrics(valid_data['actual'], valid_data['predicted'])
                    print("预测指标:", flush=True)
                    print(metrics, flush=True)
                else:
                    print("警告: 无有效数据用于计算指标", flush=True)
            except Exception as e:
                print(f"执行预测分析时出错: {e}", flush=True)
                import traceback
                traceback.print_exc(file=sys.stdout)

    # 将结果保存为JSON（如果指定了路径且有结果）
    if args.output_json and results_df is not None and isinstance(results_df, pd.DataFrame) and not results_df.empty:
        print(f"将 {args.forecast_type} 预测结果保存为JSON: {args.output_json}", flush=True)
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(args.output_json)
            if output_dir:
                 os.makedirs(output_dir, exist_ok=True)
                 print(f"确保输出目录存在: {output_dir}", flush=True)
            
            results_json = results_df.copy()
            # Ensure datetime is string
            if 'datetime' in results_json.columns:
                 results_json['datetime'] = pd.to_datetime(results_json['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Clean NaN/Inf
            for col in results_json.columns:
                if results_json[col].dtype.kind in 'fc': # Floats or complex
                    results_json[col] = results_json[col].replace([np.nan, np.inf, -np.inf], None)
            results_json = results_json.where(pd.notna(results_json), None)
            
            # Prepare final output structure
            output_data = {
                'status': "success",
                'forecast_type': args.forecast_type,
                'province': args.dataset_id,
                'start_time': results_json['datetime'].iloc[0] if 'datetime' in results_json.columns and not results_json.empty else None,
                'end_time': results_json['datetime'].iloc[-1] if 'datetime' in results_json.columns and not results_json.empty else None,
                'interval_minutes': args.interval,
                'historical_days': args.historical_days,  # 添加历史天数信息
                'weather_aware': args.weather_aware,  # 添加天气感知标识
                'weather_features': args.weather_features.split(',') if args.weather_aware and args.weather_features else None,  # 添加天气特征信息
                'renewable_enhanced': args.renewable_enhanced,  # 添加可再生能源增强标识
                'renewable_prediction_enabled': args.enable_renewable_prediction if args.renewable_enhanced else False,  # 添加新能源预测启用状态
                'renewable_types': args.renewable_types.split(',') if args.renewable_enhanced and args.renewable_types else None,  # 添加新能源类型信息
                'is_interval_forecast': args.train_prediction_type == 'interval',  # 明确标识这是区间预测
                'is_probabilistic': args.train_prediction_type == 'probabilistic', # Add flag
                'quantiles': [float(q.strip()) for q in args.quantiles.split(',')] if args.train_prediction_type == 'probabilistic' and args.quantiles else None, # Add quantiles, Handle None quantiles
                'predictions': results_json.to_dict(orient='records'),
                'metrics': metrics if metrics else {} # Include metrics if calculated
            }

            # 添加区间预测特有的统计信息
            if args.train_prediction_type == 'interval' and 'lower_bound' in results_json.columns and 'upper_bound' in results_json.columns:
                # 检查是否有interval_metrics可用
                if 'interval_metrics' in locals() and interval_metrics:
                    # 使用interval_metrics中的值
                    # 确保所有NumPy类型都被转换为Python原生类型
                    metrics_dict = {}
                    for k, v in interval_metrics.items():
                        if hasattr(v, 'item'):  # 检查是否为NumPy标量类型
                            metrics_dict[k] = v.item()  # 转换为Python原生类型
                        elif isinstance(v, np.ndarray):  # 处理NumPy数组
                            metrics_dict[k] = v.tolist()
                        else:
                            metrics_dict[k] = v
                    
                    output_data['metrics'] = metrics_dict
                    output_data['interval_statistics'] = {
                        'average_interval_width': float(interval_metrics.get('avg_interval_width')) if interval_metrics.get('avg_interval_width') is not None else None,
                        'confidence_level': float(args.confidence_level),
                        'total_predictions': int(len(results_json)),
                        'hit_rate': float(interval_metrics.get('hit_rate')) if interval_metrics.get('hit_rate') is not None else None
                    }
                else:
                    # 如果没有interval_metrics，尝试计算基本统计信息
                    avg_interval_width = (results_json['upper_bound'] - results_json['lower_bound']).mean()
                    output_data['interval_statistics'] = {
                        'average_interval_width': float(avg_interval_width) if not pd.isna(avg_interval_width) else None,
                        'confidence_level': float(args.confidence_level),
                        'total_predictions': int(len(results_json))
                    }
            
            # 确保metrics中的所有值都是JSON可序列化的
            if 'metrics' in output_data and output_data['metrics']:
                metrics_dict = {}
                for k, v in output_data['metrics'].items():
                    if hasattr(v, 'item'):  # 检查是否为NumPy标量类型
                        metrics_dict[k] = v.item()  # 转换为Python原生类型
                    elif isinstance(v, np.ndarray):  # 处理NumPy数组
                        metrics_dict[k] = v.tolist()
                    else:
                        metrics_dict[k] = v
                output_data['metrics'] = metrics_dict
                
                # 特别处理weather_scenario信息，确保它在输出中
                if 'weather_scenario' in metrics_dict:
                    print(f"正在保存weather_scenario信息到JSON输出: {type(metrics_dict['weather_scenario'])}", flush=True)
                    # 确保weather_scenario也是JSON可序列化的
                    if isinstance(metrics_dict['weather_scenario'], dict):
                        scenario_dict = {}
                        for sc_k, sc_v in metrics_dict['weather_scenario'].items():
                            if hasattr(sc_v, 'item'):
                                scenario_dict[sc_k] = sc_v.item()
                            elif isinstance(sc_v, np.ndarray):
                                scenario_dict[sc_k] = sc_v.tolist()
                            else:
                                scenario_dict[sc_k] = sc_v
                        output_data['metrics']['weather_scenario'] = scenario_dict
                    else:
                        output_data['metrics']['weather_scenario'] = metrics_dict['weather_scenario']
                else:
                    print("警告: metrics中未找到weather_scenario信息", flush=True)
                
                # 特别处理enhanced_scenario信息（可再生能源增强预测特有）
                if args.renewable_enhanced and 'enhanced_scenario' in metrics_dict:
                    print(f"正在保存enhanced_scenario信息到JSON输出: {type(metrics_dict['enhanced_scenario'])}", flush=True)
                    enhanced_scenario_dict = {}
                    for es_k, es_v in metrics_dict['enhanced_scenario'].items():
                        if hasattr(es_v, 'item'):
                            enhanced_scenario_dict[es_k] = es_v.item()
                        elif isinstance(es_v, np.ndarray):
                            enhanced_scenario_dict[es_k] = es_v.tolist()
                        elif isinstance(es_v, dict):
                            # 递归处理嵌套字典
                            nested_dict = {}
                            for nested_k, nested_v in es_v.items():
                                if hasattr(nested_v, 'item'):
                                    nested_dict[nested_k] = nested_v.item()
                                elif isinstance(nested_v, np.ndarray):
                                    nested_dict[nested_k] = nested_v.tolist()
                                else:
                                    nested_dict[nested_k] = nested_v
                            enhanced_scenario_dict[es_k] = nested_dict
                        else:
                            enhanced_scenario_dict[es_k] = es_v
                    output_data['metrics']['enhanced_scenario'] = enhanced_scenario_dict
                
                # 特别处理renewable_predictions信息（可再生能源增强预测特有）
                if args.renewable_enhanced and 'renewable_predictions' in metrics_dict:
                    print(f"正在保存renewable_predictions信息到JSON输出: {type(metrics_dict['renewable_predictions'])}", flush=True)
                    renewable_dict = {}
                    if isinstance(metrics_dict['renewable_predictions'], dict):
                        for rp_k, rp_v in metrics_dict['renewable_predictions'].items():
                            if hasattr(rp_v, 'item'):
                                renewable_dict[rp_k] = rp_v.item()
                            elif isinstance(rp_v, np.ndarray):
                                renewable_dict[rp_k] = rp_v.tolist()
                            elif isinstance(rp_v, dict):
                                # 递归处理嵌套字典
                                nested_dict = {}
                                for nested_k, nested_v in rp_v.items():
                                    if hasattr(nested_v, 'item'):
                                        nested_dict[nested_k] = nested_v.item()
                                    elif isinstance(nested_v, np.ndarray):
                                        nested_dict[nested_k] = nested_v.tolist()
                                    else:
                                        nested_dict[nested_k] = nested_v
                                renewable_dict[rp_k] = nested_dict
                            else:
                                renewable_dict[rp_k] = rp_v
                        output_data['metrics']['renewable_predictions'] = renewable_dict
                    else:
                        output_data['metrics']['renewable_predictions'] = metrics_dict['renewable_predictions']
            
            print(f"预测结果数据形状: {results_df.shape}", flush=True)
            print(f"预测结果列: {results_df.columns.tolist()}", flush=True)
            print(f"预测时间范围: {output_data['start_time']} 至 {output_data['end_time']}", flush=True)
            
            # 使用自定义JSON编码器处理NumPy类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
            print(f"{args.forecast_type} 预测结果已成功保存为JSON: {args.output_json}", flush=True)
            
        except Exception as e:
            print(f"保存JSON文件时出错: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)

if __name__ == '__main__':
    main()