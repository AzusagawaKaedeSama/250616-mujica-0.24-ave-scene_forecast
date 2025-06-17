#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
此脚本用于实现天气感知的区间预测功能。
它将利用现有的天气感知确定性预测模型，通过分析历史预测误差来生成未来的预测区间。
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import matplotlib.pyplot as plt
from scripts.forecast.day_ahead_forecast import perform_weather_aware_day_ahead_forecast

# --- 添加项目根目录到 sys.path ---
# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录 (forecast 目录)
current_script_dir = os.path.dirname(current_script_path)
# 获取 scripts 目录
scripts_dir = os.path.dirname(current_script_dir)
# 获取项目根目录 (scripts 目录的上级目录)
project_root_dir = os.path.dirname(scripts_dir)

# 将项目根目录添加到 Python 搜索路径中
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# -----------------------------------

from models.torch_models import WeatherAwareConvTransformer
from data.dataset_builder import DatasetBuilder
from utils.scaler_manager import ScalerManager

# 设置标准输出编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def get_historical_prediction_errors(province: str, forecast_type: str, history_start_date: str, history_end_date: str, seq_length: int = 96):
    """
    第一步：在历史数据上进行预测，并计算预测误差。
    
    Args:
        province (str): 省份名称, 用于定位模型和数据。
        forecast_type (str): 预测类型, 'load', 'pv', 或 'wind'。
        history_start_date (str): 用于计算历史误差的开始日期 'YYYY-MM-DD'。
        history_end_date (str): 用于计算历史误差的结束日期 'YYYY-MM-DD'。
        seq_length (int): 模型输入序列长度。

    Returns:
        list: 已排序的预测误差列表 (actual - predicted)。
    """
    print(f"--- 步骤1: 开始计算 {province} 省份, 从 {history_start_date} 到 {history_end_date} 的历史预测误差 ---")
    
    # 1. 设置路径
    model_dir = f"models/convtrans_weather/{forecast_type}/{province}"
    scaler_dir = f"models/scalers/convtrans_weather/{forecast_type}/{province}"
    data_path = f"data/timeseries_{forecast_type}_weather_{province}.csv"

    # 2. 检查路径是否存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"标准化器目录不存在: {scaler_dir}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 3. 加载模型和标准化器
    print("加载天气感知模型和标准化器...")
    try:
        model = WeatherAwareConvTransformer.load(save_dir=model_dir)
        scaler_manager = ScalerManager(scaler_path=scaler_dir)
        scaler_manager.load_scaler('X')
        scaler_manager.load_scaler('y')
    except Exception as e:
        print(f"加载模型或标准化器失败: {e}")
        return []

    # 4. 加载数据
    print(f"加载数据从: {data_path}")
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # --- 核心逻辑: 逐日进行历史回溯预测 ---
    all_errors = []
    
    date_range = pd.to_datetime(pd.date_range(start=history_start_date, end=history_end_date, freq='D'))
    print(f"将对 {len(date_range)} 天的数据进行回溯预测来收集误差...")

    for i, current_date in enumerate(date_range):
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n正在处理日期: {date_str} (进度: {i+1}/{len(date_range)})")
        
        # 定义历史数据范围，用于为当天的预测提供输入
        # 我们需要 current_date 前 seq_length 个点的数据作为输入
        history_end_for_input = current_date
        history_start_for_input = history_end_for_input - timedelta(hours=seq_length * 15 / 60) # 假设15分钟间隔
        
        historical_data = df.loc[history_start_for_input:history_end_for_input]
        
        # 检查当天是否有足够的历史数据和真实数据
        if date_str not in df.index.strftime('%Y-%m-%d').unique() or historical_data.shape[0] < seq_length:
            print(f"日期 {date_str} 的数据不完整，跳过。")
            continue

        # 使用与 `perform_weather_aware_day_ahead_forecast` 类似的方法进行单日预测
        # 注意：这里我们为了获取误差，需要调用一个能返回预测值和真实值的函数。
        # day_ahead_forecast.py 中的函数直接返回一个字典，很适合这个场景。
        # 我们需要提供必要的参数。
        
        try:
            # 模拟调用日前预测函数，获取单日预测结果
            # 这里复用这个函数，因为它内部包含了完整的预测流程：特征工程、标准化、预测、反标准化
            result_dict = perform_weather_aware_day_ahead_forecast(
                data_path=data_path,
                forecast_date=date_str,
                dataset_id=province,
                forecast_type=forecast_type,
                # 其他参数使用默认值
            )
            
            # 从返回的字典中提取预测和真实值
            if result_dict and 'predictions' in result_dict and 'actuals' in result_dict:
                predictions = result_dict['predictions']
                actuals = result_dict['actuals']
                
                # 计算误差
                day_errors = np.array(actuals) - np.array(predictions)
                
                # 过滤掉nan值
                day_errors = day_errors[~np.isnan(day_errors)]
                
                if len(day_errors) > 0:
                    all_errors.extend(day_errors)
                    print(f"日期 {date_str} 计算了 {len(day_errors)} 个误差点。")
                else:
                    print(f"日期 {date_str} 没有有效的真实值用于计算误差。")
            else:
                 print(f"日期 {date_str} 的预测未能返回有效结果。")

        except Exception as e:
            print(f"处理日期 {date_str} 时发生错误: {e}")
            import traceback
            traceback.print_exc(file=sys.stdout)

    if not all_errors:
        print("错误：未能收集到任何历史预测误差。")
        return []

    # 5. 排序并返回
    all_errors.sort()
    print(f"\n--- 步骤1完成: 共收集到 {len(all_errors)} 个历史误差样本 ---")
    
    return all_errors

def find_optimal_beta(sorted_errors: list, confidence_level: float):
    """
    第二步：根据置信水平和历史误差，寻找最优的 beta 值。
    
    Args:
        sorted_errors (list): 已排序的历史误差列表。
        confidence_level (float): 置信水平, 如 0.9, 0.95。

    Returns:
        float: 最优的 beta 值。
    """
    print(f"\n--- 步骤2: 开始寻找最优 beta (置信水平: {confidence_level}) ---")
    
    if not sorted_errors:
        print("错误：误差序列为空，无法寻找 beta。将返回默认值。")
        return (1.0 - confidence_level) / 2.0

    alpha = 1.0 - confidence_level
    
    best_beta = -1
    min_width = float('inf')

    # 在 [0, alpha] 范围内搜索最优的 beta
    # 我们以 100 个步长进行搜索，这在精度和速度之间取得了平衡
    beta_candidates = np.linspace(0, alpha, 101)
    
    print(f"将在 [0, {alpha:.2f}] 范围内搜索 {len(beta_candidates)} 个候选 beta 值...")

    for beta in beta_candidates:
        # 避免 beta + (1 - alpha) > 1 的情况
        if beta > alpha: continue

        # 计算分位数
        # quantile() 函数期望的 q 在 [0, 1] 之间
        lower_q = beta
        upper_q = 1.0 - alpha + beta
        
        # 使用 numpy.quantile 计算分位数
        lower_bound_error = np.quantile(sorted_errors, lower_q)
        upper_bound_error = np.quantile(sorted_errors, upper_q)
        
        width = upper_bound_error - lower_bound_error
        
        if width < min_width:
            min_width = width
            best_beta = beta
            
    print(f"--- 步骤2完成: 找到最优 beta = {best_beta:.4f} ---")
    print(f"   - 在此 beta 值下, 最小误差区间宽度为: {min_width:.2f}")
    
    return best_beta

def generate_weather_aware_interval_forecast(
    province: str,
    forecast_type: str,
    forecast_date: str, 
    historical_errors: list, 
    optimal_beta: float,
    confidence_level: float):
    """
    第三步：执行天气感知的区间预测。
    
    Args:
        province (str): 省份。
        forecast_type (str): 预测类型。
        forecast_date (str): 要预测的日期 'YYYY-MM-DD'。
        historical_errors (list): 已排序的历史误差序列。
        optimal_beta (float): 最优的 beta 值。
        confidence_level (float): 置信水平。

    Returns:
        pd.DataFrame: 包含预测区间的 DataFrame, 或在失败时返回空 DataFrame。
    """
    print(f"\n--- 步骤3: 开始为日期 {forecast_date} 生成天气感知区间预测 ---")
    
    alpha = 1.0 - confidence_level
    data_path = f"data/timeseries_{forecast_type}_weather_{province}.csv"

    # 1. 对目标日期进行确定性预测
    print("首先，为目标日期生成确定性点预测...")
    try:
        point_forecast_result = perform_weather_aware_day_ahead_forecast(
            data_path=data_path,
            forecast_date=forecast_date,
            dataset_id=province,
            forecast_type=forecast_type,
        )
        if not point_forecast_result or 'predictions' not in point_forecast_result:
            print(f"错误：未能获取日期 {forecast_date} 的确定性预测结果。")
            return pd.DataFrame()

        # 将结果转换为 DataFrame
        forecast_df = pd.DataFrame({
            'datetime': point_forecast_result['timestamps'],
            'predicted': point_forecast_result['predictions'],
            'actual': point_forecast_result.get('actuals', None) # actual 可能不存在
        })
        forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])
        print(f"已成功获取 {len(forecast_df)} 个点的确定性预测。")
        
    except Exception as e:
        print(f"为日期 {forecast_date} 进行确定性预测时失败: {e}")
        return pd.DataFrame()

    # 2. 计算误差分位数
    if not historical_errors:
        print("警告: 历史误差序列为空，无法计算区间。")
        return forecast_df # 至少返回点预测

    lower_q = optimal_beta
    upper_q = 1.0 - alpha + optimal_beta
    
    error_lower_bound = np.quantile(historical_errors, lower_q)
    error_upper_bound = np.quantile(historical_errors, upper_q)
    
    print(f"使用最优 beta={optimal_beta:.4f} 计算误差边界:")
    print(f" - 误差下界 ({lower_q*100:.1f}% 分位): {error_lower_bound:.2f}")
    print(f" - 误差上界 ({upper_q*100:.1f}% 分位): {error_upper_bound:.2f}")

    # 3. 生成预测区间
    forecast_df['lower_bound'] = forecast_df['predicted'] + error_lower_bound
    forecast_df['upper_bound'] = forecast_df['predicted'] + error_upper_bound
    
    print("--- 步骤3完成: 预测区间已生成 ---")
    
    return forecast_df

def plot_interval_forecast_results(results_df: pd.DataFrame, province: str, forecast_date: str, confidence_level: float):
    """
    为天气感知区间预测结果绘制一张图表。

    Args:
        results_df (pd.DataFrame): 包含预测结果的DataFrame。
        province (str): 省份名称。
        forecast_date (str): 预测日期。
        confidence_level (float): 置信水平，用于图表标签。
    """
    # --- 新增：设置中文字体，解决OSError ---
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    except Exception as e:
        print(f"设置中文字体失败，图表中的中文可能无法正常显示: {e}")
    # ------------------------------------
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))

    # 绘制实际值 (如果存在)
    if 'actual' in results_df.columns and not results_df['actual'].dropna().empty:
        ax.plot(results_df['datetime'], results_df['actual'], label='Actual Value', color='red', linewidth=2, zorder=5)

    # 绘制中心预测值
    ax.plot(results_df['datetime'], results_df['predicted'], label='Predicted Value (Centerline)', color='blue', linestyle='--', linewidth=2, zorder=4)

    # 填充预测区间
    ax.fill_between(
        results_df['datetime'],
        results_df['lower_bound'],
        results_df['upper_bound'],
        color='blue',
        alpha=0.2,
        label=f'{int(confidence_level*100)}% Prediction Interval'
    )

    # --- 美化图表 ---
    ax.set_title(f'Weather-Aware Interval Forecast for {province} on {forecast_date}', fontsize=18)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Load Value', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 格式化x轴日期显示
    fig.autofmt_xdate()
    
    plt.tight_layout()

    # --- 保存图表 ---
    save_dir = os.path.join('results', 'weather_aware_interval', province)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f"interval_forecast_{forecast_date}_{timestamp}.png")
    
    try:
        plt.savefig(save_path, dpi=300)
        print(f"\n预测结果图表已成功保存至: {save_path}")
    except Exception as e:
        print(f"\n保存图表时出错: {e}")
    finally:
        plt.close(fig)

def perform_weather_aware_interval_forecast_for_range(
    province: str,
    forecast_type: str,
    start_date_str: str,
    end_date_str: str,
    confidence_level: float,
    historical_days: int,
    **kwargs
):
    """
    为指定日期范围执行天气感知区间预测。
    此函数协调整个流程：收集历史误差、找到最优区间参数、为每个日期生成预测。

    Args:
        province (str): 省份名称.
        forecast_type (str): 预测类型 ('load', 'pv', 或 'wind').
        start_date_str (str): 预测开始日期 'YYYY-MM-DD'.
        end_date_str (str): 预测结束日期 'YYYY-MM-DD'.
        confidence_level (float): 区间置信水平.
        historical_days (int): 用于计算历史误差的天数.
        **kwargs: 备用参数.

    Returns:
        tuple: 包含预测结果的 DataFrame 和一个包含评估指标的字典。
    """
    print(f"\n--- 开始为 {province} 从 {start_date_str} 到 {end_date_str} 执行天气感知区间预测 ---")
    
    # 1. 确定历史误差计算周期
    forecast_start_date = pd.to_datetime(start_date_str)
    history_end_date = forecast_start_date - timedelta(days=1)
    history_start_date = history_end_date - timedelta(days=historical_days - 1)
    
    history_start_date_str = history_start_date.strftime('%Y-%m-%d')
    history_end_date_str = history_end_date.strftime('%Y-%m-%d')

    # 2. 获取历史误差
    historical_errors = get_historical_prediction_errors(
        province=province,
        forecast_type=forecast_type,
        history_start_date=history_start_date_str,
        history_end_date=history_end_date_str,
    )

    if not historical_errors:
        print(f"错误：未能为省份 {province} 收集历史预测误差。中止操作。")
        return pd.DataFrame(), {}

    # 3. 寻找最优 beta
    optimal_beta = find_optimal_beta(
        sorted_errors=historical_errors,
        confidence_level=confidence_level
    )

    # 4. 循环预测范围内的每一天
    all_results_df = []
    date_range = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
    
    print(f"\n--- 开始为 {len(date_range)} 天生成区间预测 ---")
    for forecast_date in date_range:
        date_str = forecast_date.strftime('%Y-%m-%d')
        
        daily_results_df = generate_weather_aware_interval_forecast(
            province=province,
            forecast_type=forecast_type,
            forecast_date=date_str,
            historical_errors=historical_errors,
            optimal_beta=optimal_beta,
            confidence_level=confidence_level
        )
        
        if not daily_results_df.empty:
            daily_results_df['prediction_date'] = date_str
            all_results_df.append(daily_results_df)
        else:
            print(f"警告：未能为日期 {date_str} 生成预测结果。")

    if not all_results_df:
        print("错误：未能生成任何预测结果。")
        return pd.DataFrame(), {}

    # 5. 合并结果并计算指标
    final_df = pd.concat(all_results_df, ignore_index=True)
    
    metrics = {}
    
    # 5.1 添加天气场景分析和典型场景匹配
    weather_scenario_info = {}
    if forecast_type in ['load', 'pv', 'wind'] and not final_df.empty:
        try:
            print("\n--- 开始天气场景分析和典型场景匹配 ---")
            
            # 加载天气数据文件
            weather_data_path = f"data/timeseries_{forecast_type}_weather_{province}.csv"
            weather_data = pd.read_csv(weather_data_path, index_col=0)
            weather_data.index = pd.to_datetime(weather_data.index)
            
            # 检查预测起始日期的天气数据
            forecast_start_date = pd.to_datetime(start_date_str)
            forecast_day_data = weather_data[weather_data.index.date == forecast_start_date.date()]
            
            if not forecast_day_data.empty:
                # 进行天气场景分析 - 使用增强的场景库
                from utils.enhanced_scenario_library import create_enhanced_scenario_classifier
                classifier = create_enhanced_scenario_classifier()
                
                # 准备天气数据列名
                weather_columns_mapping = {
                    'weather_temperature_c': 'temperature',
                    'weather_wind_speed': 'wind_speed', 
                    'weather_relative_humidity': 'humidity',
                    'weather_precipitation_mm': 'precipitation'
                }
                
                weather_data_renamed = forecast_day_data.copy()
                for old_name, new_name in weather_columns_mapping.items():
                    if old_name in weather_data_renamed.columns:
                        weather_data_renamed[new_name] = weather_data_renamed[old_name]
                
                # 分析天气场景
                scenario_result = classifier.identify_scenario(weather_data_renamed)
                weather_scenario_info = {
                    'scenario_type': scenario_result['name'],
                    'temperature_mean': weather_data_renamed.get('temperature', forecast_day_data.get('weather_temperature_c', pd.Series([25.0]))).mean(),
                    'temperature_max': weather_data_renamed.get('temperature', forecast_day_data.get('weather_temperature_c', pd.Series([30.0]))).max(),
                    'temperature_min': weather_data_renamed.get('temperature', forecast_day_data.get('weather_temperature_c', pd.Series([20.0]))).min(),
                    'humidity_mean': weather_data_renamed.get('humidity', forecast_day_data.get('weather_relative_humidity', pd.Series([60.0]))).mean(),
                    'wind_speed_mean': weather_data_renamed.get('wind_speed', forecast_day_data.get('weather_wind_speed', pd.Series([3.0]))).mean(),
                    'precipitation_sum': weather_data_renamed.get('precipitation', forecast_day_data.get('weather_precipitation_mm', pd.Series([0.0]))).sum()
                }
                print(f"天气场景分析结果: {scenario_result['name']}")
                
                # 进行典型场景匹配
                from utils.scenario_matcher import ScenarioMatcher
                matcher = ScenarioMatcher(province)
                
                # 计算当日负荷特征
                first_day_data = final_df[final_df['datetime'].dt.date == forecast_start_date.date()]
                if not first_day_data.empty:
                    daily_load_mean = first_day_data['predicted'].mean()
                    daily_load_volatility = first_day_data['predicted'].std() / daily_load_mean if daily_load_mean > 0 else 0.15
                    
                    # 构建特征向量
                    current_features = {
                        'temperature_mean': weather_scenario_info['temperature_mean'],
                        'humidity_mean': weather_scenario_info['humidity_mean'],
                        'wind_speed_mean': weather_scenario_info['wind_speed_mean'],
                        'precipitation_sum': weather_scenario_info['precipitation_sum'],
                        'load_mean': daily_load_mean,
                        'load_volatility': daily_load_volatility
                    }
                    
                    # 执行场景匹配
                    match_result = matcher.match_scenario(current_features, province)
                    
                    if match_result:
                        scenario_match_info = {
                            'matched_scenario': match_result['matched_scenario']['name'],
                            'similarity': match_result['matched_scenario']['similarity'],
                            'similarity_percentage': match_result['matched_scenario']['similarity_percentage'],
                            'confidence_level': match_result['confidence_level'],
                            'description': match_result['matched_scenario']['description'],
                            'typical_percentage': match_result['matched_scenario']['typical_percentage'],
                            'distance': match_result['matched_scenario']['distance'],
                            'top_scenarios': [
                                {
                                    'name': scenario['name'],
                                    'similarity_percentage': scenario['similarity_percentage'],
                                    'rank': scenario['rank']
                                }
                                for scenario in match_result['all_scenarios'][:3]
                            ],
                            'feature_contributions': match_result['feature_analysis']
                        }
                        
                        # 将场景匹配信息添加到天气场景信息中
                        weather_scenario_info.update({
                            'scenario_match': scenario_match_info,
                            'daily_load_mean': daily_load_mean,
                            'daily_load_volatility': daily_load_volatility
                        })
                        
                        print(f"典型场景匹配结果: {scenario_match_info['matched_scenario']} "
                              f"(相似度: {scenario_match_info['similarity_percentage']:.1f}%)")
                        
                        # ========== 新增：场景感知的不确定度调整 ==========
                        print("\n--- 开始应用场景感知的不确定度调整 ---")
                        
                        # 获取场景的不确定度倍数
                        uncertainty_multiplier = scenario_result.get('uncertainty_multiplier', 1.0)
                        print(f"识别场景: {scenario_result['name']}")
                        print(f"不确定度倍数: {uncertainty_multiplier}")
                        
                        # 根据场景置信度进一步调整
                        scenario_confidence = match_result['confidence_level']
                        if scenario_confidence < 0.3:  # 低置信度场景
                            confidence_adjustment = 1.2  # 增加20%不确定度
                            print(f"低场景置信度 ({scenario_confidence:.2f})，额外增加不确定度: {confidence_adjustment}")
                        elif scenario_confidence > 0.7:  # 高置信度场景
                            confidence_adjustment = 0.9   # 降低10%不确定度
                            print(f"高场景置信度 ({scenario_confidence:.2f})，适度降低不确定度: {confidence_adjustment}")
                        else:
                            confidence_adjustment = 1.0   # 保持不变
                            
                        # 计算最终调整因子
                        final_adjustment_factor = uncertainty_multiplier * confidence_adjustment
                        print(f"最终不确定度调整因子: {final_adjustment_factor:.2f}")
                        
                        # 应用到所有预测点的区间
                        if 'lower_bound' in final_df.columns and 'upper_bound' in final_df.columns:
                            print("正在调整预测区间...")
                            
                            for idx in range(len(final_df)):
                                # 获取点预测值
                                predicted_value = final_df.iloc[idx]['predicted']
                                
                                # 计算原始区间半宽度
                                original_lower = final_df.iloc[idx]['lower_bound']
                                original_upper = final_df.iloc[idx]['upper_bound']
                                original_half_width = (original_upper - original_lower) / 2
                                
                                # 应用场景调整
                                adjusted_half_width = original_half_width * final_adjustment_factor
                                
                                # 根据时段进一步微调（高峰时段增加不确定度）
                                hour = final_df.iloc[idx]['datetime'].hour
                                if 18 <= hour <= 21:  # 晚高峰
                                    time_adjustment = 1.1
                                elif 7 <= hour <= 9:   # 早高峰
                                    time_adjustment = 1.05
                                elif 0 <= hour <= 6:   # 深夜低谷
                                    time_adjustment = 0.95
                                else:
                                    time_adjustment = 1.0
                                    
                                final_half_width = adjusted_half_width * time_adjustment
                                
                                # 更新区间边界
                                final_df.iloc[idx, final_df.columns.get_loc('lower_bound')] = max(0, predicted_value - final_half_width)
                                final_df.iloc[idx, final_df.columns.get_loc('upper_bound')] = predicted_value + final_half_width
                            
                            # 计算调整后的平均区间宽度
                            adjusted_avg_width = (final_df['upper_bound'] - final_df['lower_bound']).mean()
                            print(f"场景调整后平均区间宽度: {adjusted_avg_width:.2f}")
                            
                            # 记录调整信息到weather_scenario_info
                            weather_scenario_info['uncertainty_adjustment'] = {
                                'scenario_multiplier': uncertainty_multiplier,
                                'confidence_adjustment': confidence_adjustment,
                                'final_adjustment_factor': final_adjustment_factor,
                                'adjusted_avg_width': adjusted_avg_width,
                                'adjustment_method': 'weather_scenario_aware'
                            }
                        # ================= 场景感知不确定度调整结束 =================
                        
                    else:
                        print("警告: 典型场景匹配失败")
                        weather_scenario_info['scenario_match_error'] = "场景匹配失败"
                        
            else:
                print(f"警告: 在天气数据中未找到预测日 {start_date_str} 的数据，使用默认值")
                weather_scenario_info = {
                    'scenario_type': '数据缺失',
                    'temperature_mean': 25.0,
                    'temperature_max': 30.0,
                    'temperature_min': 20.0,
                    'humidity_mean': 60.0,
                    'wind_speed_mean': 3.0,
                    'precipitation_sum': 0.0
                }
                
        except Exception as e:
            print(f"警告: 天气场景分析失败: {e}")
            weather_scenario_info = {'error': str(e)}
    
    # 5.2 计算基本预测指标
    if 'actual' in final_df.columns and final_df['actual'].notna().any():
        # 导入评估函数
        from utils.evaluator import calculate_metrics

        valid_data = final_df.dropna(subset=['actual', 'predicted', 'lower_bound', 'upper_bound'])
        if not valid_data.empty:
            # 点预测指标
            metrics = calculate_metrics(valid_data['actual'], valid_data['predicted'])
            
            # 区间预测指标
            hits = ((valid_data['actual'] >= valid_data['lower_bound']) & 
                    (valid_data['actual'] <= valid_data['upper_bound']))
            metrics['hit_rate'] = hits.mean()
            metrics['avg_interval_width'] = (valid_data['upper_bound'] - valid_data['lower_bound']).mean()
            
            print("\n--- 综合预测指标 ---")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
    
    # 5.3 添加天气场景信息到metrics中
    if weather_scenario_info:
        metrics['weather_scenario'] = weather_scenario_info

    # 为第一天的结果绘图
    if not final_df.empty:
        first_day_df = final_df[final_df['datetime'].dt.date == pd.to_datetime(start_date_str).date()].copy()
        if not first_day_df.empty:
            plot_interval_forecast_results(
                results_df=first_day_df,
                province=province,
                forecast_date=start_date_str,
                confidence_level=confidence_level
            )

    return final_df, metrics

if __name__ == '__main__':
    """
    主执行块，用于测试此脚本中的功能。
    """
    print("--- 天气感知区间预测脚本 ---")
    
    # --- 测试步骤1 ---
    # 定义测试参数
    test_province = '上海'
    test_forecast_type = 'load'  # 可以改为 'pv' 或 'wind' 进行测试
    # --- 使用用户建议的固定日期 ---
    test_history_start_date = '2024-01-01'
    test_history_end_date = '2024-01-31'# 为了加快速度，现在这样设置
    test_forecast_date = '2024-02-01'
    # ---------------------------

    # 调用函数
    historical_errors = get_historical_prediction_errors(
        province=test_province,
        forecast_type=test_forecast_type,
        history_start_date=test_history_start_date,
        history_end_date=test_history_end_date
    )

    # 打印结果进行验证
    if historical_errors:
        print("\n--- 历史误差序列计算结果 (部分展示) ---")
        print(f"总误差数量: {len(historical_errors)}")
        print(f"误差最小值 (预测最偏高): {historical_errors[0]:.2f}")
        print(f"误差最大值 (预测最偏低): {historical_errors[-1]:.2f}")
        
        # 打印一些分位数点
        p10 = np.percentile(historical_errors, 10)
        p50 = np.percentile(historical_errors, 50) # 中位数
        p90 = np.percentile(historical_errors, 90)
        print(f"10% 分位数: {p10:.2f}")
        print(f"50% 分位数 (中位数): {p50:.2f}")
        print(f"90% 分位数: {p90:.2f}")
        
        # --- 测试步骤2 ---
        test_confidence_level = 0.90
        optimal_beta = find_optimal_beta(
            sorted_errors=historical_errors,
            confidence_level=test_confidence_level
        )
        print(f"\n--- 最优 beta 计算结果 ---")
        print(f"在 {test_confidence_level*100}% 置信水平下, 计算出的最优 beta 为: {optimal_beta:.4f}")

        # --- 测试步骤3 ---
        interval_results_df = generate_weather_aware_interval_forecast(
            province=test_province,
            forecast_type=test_forecast_type,
            forecast_date=test_forecast_date,
            historical_errors=historical_errors,
            optimal_beta=optimal_beta,
            confidence_level=test_confidence_level
        )

        if not interval_results_df.empty:
            print(f"\n--- 天气感知区间预测最终结果 (日期: {test_forecast_date}) ---")
            # 使用 .to_string() 保证所有列和行都显示
            print(interval_results_df.to_string())

            # --- 新增：调用绘图函数 ---
            plot_interval_forecast_results(
                results_df=interval_results_df,
                province=test_province,
                forecast_date=test_forecast_date,
                confidence_level=test_confidence_level
            )
            # ---------------------------
        else:
            print(f"\n--- 未能生成区间预测结果 ---")

    else:
        print("\n--- 历史误差序列计算失败 ---")
    
    print("\n--- 脚本执行完毕 ---")