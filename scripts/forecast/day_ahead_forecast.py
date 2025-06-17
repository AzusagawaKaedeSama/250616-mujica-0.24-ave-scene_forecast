# 日前预测函数 - 使用 DatasetBuilder
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from utils.evaluator import calculate_metrics
from models.torch_models import PeakAwareConvTransformer
from utils.scaler_manager import ScalerManager
from data.dataset_builder import DatasetBuilder

# 设置标准输出编码为UTF-8，解决中文显示乱码问题
sys.stdout.reconfigure(encoding='utf-8')

def perform_day_ahead_forecast(data_path, forecast_date, peak_hours=(8, 22), valley_hours=(0, 7), 
                             peak_weight=2.5, valley_weight=1.5, dataset_id="上海", forecast_type="load",
                             historical_days=8, forecast_end_date=None):
    """
    执行日前预测功能，预测指定日期范围的数据
    
    Args:
        data_path: 数据路径
        forecast_date: 要预测的开始日期（字符串，格式：'YYYY-MM-DD'）
        peak_hours: 高峰时段起止小时，默认为(8, 22)
        valley_hours: 低谷时段起止小时，默认为(0, 7)
        peak_weight: 高峰时段权重，默认为2.5
        valley_weight: 低谷时段权重，默认为1.5
        dataset_id: 数据集标识，默认为"上海"
        forecast_type: 预测类型，可以是'load'、'pv'或'wind'，默认为'load'
        historical_days: 使用的历史数据天数，默认为8天
        forecast_end_date: 要预测的结束日期（字符串，格式：'YYYY-MM-DD'），默认为None（单日预测）
        
    Returns:
        DataFrame: 包含预测结果的DataFrame
    """
    # 根据预测类型确定值列名
    value_column = forecast_type
    
    # 处理日期范围
    if forecast_end_date is None:
        # 如果未提供结束日期，则只预测一天
        forecast_end_date = forecast_date
    
    try:
        # 解析预测日期
        start_day = pd.to_datetime(forecast_date)
        end_day = pd.to_datetime(forecast_end_date)
        
        # 确保开始日期不晚于结束日期
        if start_day > end_day:
            print(f"错误: 开始日期 {forecast_date} 晚于结束日期 {forecast_end_date}", flush=True)
            return None
        
        date_range = pd.date_range(start=start_day, end=end_day, freq='D')
        total_days = len(date_range)
        
        print(f"\n=== 执行日前{forecast_type}预测：{forecast_date} 至 {forecast_end_date} ({dataset_id}) ===", flush=True)
        print(f"总计预测 {total_days} 天，使用 {historical_days} 天的历史数据", flush=True)
        
    except Exception as e:
        print(f"日期格式错误: {e}，应为'YYYY-MM-DD'格式", flush=True)
        return None
    
    # --- 更新: 使用包含 forecast_type 的路径 --- 
    # 日前预测通常使用高峰感知模型
    model_base_name = 'convtrans_peak' 
    model_dir = f"models/{model_base_name}/{forecast_type}/{dataset_id}"
    results_dir = f"results/day_ahead/{model_base_name}/{forecast_type}/{dataset_id}" # <-- 使用 model_base_name
    scaler_dir = f"models/scalers/{model_base_name}/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)
    # -------------------------------------------
    
    # 检查模型和缩放器是否存在
    if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir)):
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在或其中没有 .pth 模型文件")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"缩放器目录不存在: {scaler_dir}")
    
    print(f"从 {model_dir} 加载已训练的模型...", flush=True)
    model = PeakAwareConvTransformer.load(save_dir=model_dir)
    
    # 从模型配置中获取seq_length，确保与训练时一致
    seq_length = model.config.get('seq_length', 96)
    interval_minutes = int(1440 / seq_length) if seq_length > 0 else 15 # 推断间隔
    print(f"从模型配置中获取seq_length: {seq_length} (推断间隔: {interval_minutes} 分钟)", flush=True)
    
    print(f"从 {scaler_dir} 加载缩放器...", flush=True)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    # 加载X和y标准化器
    scaler_manager.load_scaler('X')
    scaler_manager.load_scaler('y')
    
    # 加载时间序列数据
    ts_data_path = data_path
    if not os.path.exists(ts_data_path):
        raise FileNotFoundError(f"时间序列数据文件不存在: {ts_data_path}")

    print(f"从 {ts_data_path} 加载时间序列数据...", flush=True)
    ts_data = pd.read_csv(ts_data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # 初始化数据集构建器
    dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1)
    
    # 初始化结果DataFrame列表，用于存储每一天的预测结果
    all_results = []
    
    # 获取并显示一天的预测点数
    points_per_day = int(24 * 60 / interval_minutes)
    print(f"每天将预测 {points_per_day} 个时间点", flush=True)
    
    # 遍历日期范围，逐日预测
    for i, current_day in enumerate(date_range):
        current_date_str = current_day.strftime('%Y-%m-%d')
        print(f"\n--- 正在预测第 {i+1}/{total_days} 天: {current_date_str} ---", flush=True)
        
        # --- 获取历史数据，基于指定的历史天数 --- 
        historical_end_dt = current_day # 当前预测日前一天的23:59:59
        
        # 获取足够的历史数据以覆盖seq_length点和额外的历史天数
        # 1. 计算基于seq_length所需的最小历史时间跨度
        min_history_start_dt = historical_end_dt - timedelta(minutes=seq_length * interval_minutes)
        
        # 2. 计算基于历史天数的开始时间
        days_based_start_dt = historical_end_dt - timedelta(days=historical_days)
        
        # 3. 选择更早的时间点作为最终的历史数据开始点
        historical_start_dt = min(min_history_start_dt, days_based_start_dt)
        
        print(f"获取历史数据，从 {historical_start_dt} 到 {historical_end_dt}", flush=True)
        historical_data = ts_data[(ts_data.index >= historical_start_dt) & (ts_data.index < historical_end_dt)].copy()
        
        if len(historical_data) < seq_length:
            print(f"警告: 历史数据不足: 从 {historical_start_dt} 到 {historical_end_dt} 仅有 {len(historical_data)} 个数据点，需要 {seq_length} 个", flush=True)
            # 对于多天预测，我们可以跳过无法预测的天，继续预测下一天
            if len(date_range) > 1:
                print(f"跳过 {current_date_str} 的预测，继续下一天", flush=True)
                continue
            else:
                # 如果是单天预测，则仍然报错
                raise ValueError("历史数据不足，无法进行日前预测", flush=True)
        
        # 准备预测时间点
        forecast_start = pd.Timestamp(f"{current_date_str} 00:00:00")
        forecast_end = pd.Timestamp(f"{current_date_str} 23:59:59") # 确保包含全天
        forecast_times = pd.date_range(start=forecast_start, end=forecast_end, freq=f'{interval_minutes}min')
        num_predictions = len(forecast_times)
        print(f"将预测 {num_predictions} 个时间点 (从 {forecast_times[0]} 到 {forecast_times[-1]})" , flush=True)

        # --- 日前预测的输入准备方式 --- 
        # 使用最近的 seq_length 个历史点来预测未来一天的点
        
        # 构建增强数据集 (只基于已知的历史数据)
        print("对历史数据进行特征工程...", flush=True)
        enhanced_history = dataset_builder.build_dataset_with_peak_awareness(
            df=historical_data, # 仅使用历史数据
            date_column=None,  # 已经是索引
            value_column=value_column,
            interval=interval_minutes,
            peak_hours=peak_hours,
            valley_hours=valley_hours,
            peak_weight=peak_weight,
            valley_weight=valley_weight
        )
                
        # 提取模型输入 X
        X = enhanced_history.drop(columns=[value_column]).values
        if X.shape[0] != seq_length:
            #  print(f"警告: 提取的历史特征数据点数 ({X.shape[0]}) 与 seq_length ({seq_length}) 不匹配", flush=True)
             # 取最新的seq_length个点
             X = X[-seq_length:]
        
        X = X.reshape(1, seq_length, -1) # [1, seq, features]
        print(f"准备好的模型输入形状: {X.shape}", flush=True)
        
        # 标准化输入数据
        print("标准化输入数据...", flush=True)
        X_scaled = scaler_manager.transform('X', X.reshape(1, -1)).reshape(X.shape)
        
        # 初始化当前日期的预测结果
        final_predictions = None
        
        # --- 实现递归滚动预测 ---
        try:
            print(f"开始递归滚动预测 {num_predictions} 个时间点...", flush=True)
            
            # 初始化预测结果数组
            final_predictions = np.zeros(num_predictions)
            
            # 初始化当前历史数据，用于递归更新
            current_history = historical_data.copy()
            
            # 递归滚动预测
            for j in range(num_predictions):
                # 当前预测时间点
                current_time = forecast_times[j]
                
                if j % 20 == 0:  # 每20次预测显示一次进度
                    print(f"正在预测 {j+1}/{num_predictions}: {current_time}", flush=True)
                
                # 构建增强数据集
                try:
                    # 确保使用所有可用的历史数据
                    enhanced_current = dataset_builder.build_dataset_with_peak_awareness(
                        df=current_history,
                        date_column=None,  # 已经是索引
                        value_column=value_column,
                        interval=interval_minutes,
                        peak_hours=peak_hours,
                        valley_hours=valley_hours,
                        peak_weight=peak_weight,
                        valley_weight=valley_weight
                    )
                    
                    # 提取模型输入 X
                    X_current = enhanced_current.drop(columns=[value_column]).values
                    
                    # 验证数据长度
                    if X_current.shape[0] != seq_length:
                        # if j % 20 == 0:  # 减少日志输出频率
                        #     print(f"警告: 特征数据长度 ({X_current.shape[0]}) 与所需长度 ({seq_length}) 不匹配", flush=True)
                        if X_current.shape[0] < seq_length:
                            pad_length = seq_length - X_current.shape[0]
                            X_current = np.pad(X_current, ((0, pad_length), (0, 0)), 'edge')
                        else:
                            X_current = X_current[-seq_length:]
                    
                    # 重塑为模型输入形状
                    X_current = X_current.reshape(1, seq_length, -1)  # [1, seq_length, features]
                    
                    # 标准化输入 - 修复维度问题，使用与初始标准化相同的方式
                    # 将3D输入展平为2D进行标准化，然后重塑回3D
                    X_current_scaled = scaler_manager.transform('X', X_current.reshape(1, -1)).reshape(X_current.shape)
                    
                    # 使用模型预测
                    raw_pred = model.predict(X_current_scaled)
                    
                    # 提取预测值
                    if isinstance(raw_pred, np.ndarray):
                        if raw_pred.ndim > 1:
                            pred_value = raw_pred[0, 0]  # 取第一个样本的第一个预测值
                        else:
                            pred_value = raw_pred[0]  # 只有一个预测值
                    else:
                        pred_value = raw_pred
                    
                    # 反标准化得到实际预测值
                    predicted_value = scaler_manager.inverse_transform('y', np.array([[pred_value]]))
                    final_predictions[j] = predicted_value.item()
                    
                    # 检查预测值是否为NaN
                    if np.isnan(final_predictions[j]):
                        raise ValueError("预测结果为NaN")
                
                except Exception as e:
                    if j % 20 == 0:  # 减少日志输出频率
                        print(f"时间点 {j} 预测失败: {e}", flush=True)
                    # 使用备选策略
                    if j == 0:
                        # 第一个点使用历史数据最后一个值
                        final_predictions[j] = current_history[value_column].iloc[-1]
                    else:
                        # 其他点使用前一个预测值
                        final_predictions[j] = final_predictions[j-1]
                
                # 更新历史数据用于下一次预测（如果不是最后一个点）
                if j < num_predictions - 1:
                    # 创建包含新预测的数据点
                    new_data_point = pd.DataFrame(
                        {value_column: [final_predictions[j]]}, 
                        index=[current_time]
                    )
                    
                    # 更新历史数据: 移除最早一个点，添加新预测点
                    current_history = current_history.iloc[1:].copy()  # 删除最早的一个点
                    current_history = pd.concat([current_history, new_data_point])  # 添加新预测
            
            # 检查预测结果质量
            nan_count = np.isnan(final_predictions).sum()
            zero_count = np.sum(final_predictions == 0)
            if nan_count > 0:
                print(f"警告: 预测结果中有 {nan_count} 个NaN值", flush=True)
                # 替换NaN值
                hist_mean = historical_data[value_column].mean()
                final_predictions = np.nan_to_num(final_predictions, nan=hist_mean)
            
            if zero_count > num_predictions * 0.5:  # 如果超过一半是0值
                print(f"警告: 预测结果中有 {zero_count}/{num_predictions} 个零值，可能预测质量不佳", flush=True)
            
            print(f"递归预测完成，最终结果: 最小值={np.min(final_predictions):.2f}, 最大值={np.max(final_predictions):.2f}, 平均值={np.mean(final_predictions):.2f}")
                
        except Exception as e:
            print(f"递归滚动预测失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            
            # 对于多天预测，我们应该继续下一天而不是直接返回None
            if len(date_range) > 1:
                print(f"跳过 {current_date_str} 的预测，继续下一天", flush=True)
                continue
            else:
                return None  # 单天预测，直接返回None表示失败
    
        # 创建当天的结果 DataFrame
        day_results_df = pd.DataFrame({
            'datetime': forecast_times,
            'predicted': final_predictions
        })
        
        # 添加实际值（如果有）
        actual_data = ts_data[(ts_data.index >= forecast_start) & (ts_data.index <= forecast_end)].copy()
        if len(actual_data) > 0:
            day_results_df = day_results_df.set_index('datetime')
            day_results_df['actual'] = None  # 初始化为None
            # 使用 reindex 和 fillna 可能更高效
            actual_series = actual_data[value_column].reindex(day_results_df.index)
            day_results_df['actual'] = actual_series
            day_results_df = day_results_df.reset_index()
        
        # 添加高峰时段标记
        day_results_df['is_peak'] = day_results_df['datetime'].apply(
            lambda x: peak_hours[0] <= x.hour <= peak_hours[1]
        ).astype(int)
        
        # # 计算性能指标（如果有实际值）
        # if 'actual' in day_results_df.columns and not day_results_df['actual'].isna().all():
        #     valid_df = day_results_df.dropna(subset=['actual', 'predicted'])
        #     if len(valid_df) > 0:
        #         metrics = calculate_metrics(valid_df['actual'].values, valid_df['predicted'].values)
        #         print(f"\n{current_date_str} 日前预测性能指标:")
        #         for metric, value in metrics.items(): print(f"  {metric}: {value:.4f}")
                
        #         # 计算高峰时段和非高峰时段的指标
        #         peak_df = valid_df[valid_df['is_peak'] == 1]
        #         non_peak_df = valid_df[valid_df['is_peak'] == 0]
                
        #         if len(peak_df) > 0:
        #             peak_metrics = calculate_metrics(peak_df['actual'].values, peak_df['predicted'].values)
        #             print(f"\n{current_date_str} 高峰时段预测性能:")
        #             for metric, value in peak_metrics.items(): print(f"  {metric}: {value:.4f}")
                
        #         if len(non_peak_df) > 0:
        #             non_peak_metrics = calculate_metrics(non_peak_df['actual'].values, non_peak_df['predicted'].values)
        #             print(f"\n{current_date_str} 非高峰时段预测性能:")
        #             for metric, value in non_peak_metrics.items(): print(f"  {metric}: {value:.4f}")
        
        # 如果是多天预测，为每天的预测结果添加日期标识
        day_results_df['prediction_date'] = current_date_str
        
        # 添加到总结果列表
        all_results.append(day_results_df)
        
        # 为多天预测创建单独的每日可视化
        if len(date_range) > 1:
            plt.figure(figsize=(12, 6))
            
            # 绘制实际值和预测值
            plt.plot(day_results_df.index, day_results_df['predicted'], label='预测值', color='blue', marker='o', markersize=4)
            if 'actual' in day_results_df.columns and not day_results_df['actual'].isna().all():
                plt.plot(day_results_df.index, day_results_df['actual'], label='实际值', color='red', marker='x', markersize=4)
            
            # 标记高峰时段
            peak_hours_indices = day_results_df[day_results_df['is_peak'] == 1].index.tolist()
            for idx in peak_hours_indices:
                plt.axvspan(idx-0.5, idx+0.5, color='yellow', alpha=0.3)
            
            # # 添加标题和标签
            # plt.title(f"日前{forecast_type.upper()}预测 - {current_date_str} ({dataset_id})")
            # plt.xlabel('时间点')
            # plt.ylabel(f'{forecast_type.capitalize()} ({value_column})')
            # plt.legend()
            # plt.grid(True)
            
            # # 保存图表 (使用更新后的 results_dir)
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # day_plot_path = f"{results_dir}/day_ahead_forecast_{current_date_str}_{timestamp}.png"
            # plt.tight_layout()
            # plt.savefig(day_plot_path, dpi=300)
            # print(f"{current_date_str} 预测图表已保存至 {day_plot_path}")
            # plt.close() # 关闭图表，释放内存
    
    # --- 最终结果处理 ---
    if not all_results:
        print("错误: 未能生成任何预测结果。", flush=True)
        return None

    # 合并所有日期的预测结果
    final_results_df = pd.concat(all_results, ignore_index=True)
    
    # 根据需要对合并后的结果进行排序
    if 'datetime' in final_results_df.columns:
        final_results_df = final_results_df.sort_values(by='datetime').reset_index(drop=True)
    
    # 保存最终的合并结果 (可选，但建议) (使用更新后的 results_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_csv_path = f"{results_dir}/day_ahead_forecast_{start_day.strftime('%Y%m%d')}_{end_day.strftime('%Y%m%d')}_{timestamp}.csv"
    try:
        final_results_df.to_csv(final_csv_path, index=False)
        # print(f"所有日期的日前预测结果已保存至 {final_csv_path}")
    except Exception as e:
        print(f"保存最终合并的CSV文件时出错: {e}", flush=True)
        return None
        
    return final_results_df # 返回包含所有天数结果的DataFrame

def perform_day_ahead_forecast_with_smooth_transition(
    data_path,
    forecast_date,
    peak_hours=(8, 20),
    valley_hours=(0, 7),
    peak_weight=2.5,
    valley_weight=1.5,
    dataset_id='上海',
    max_allowed_diff_pct=5.0,  # 允许的最大百分比差异，可调整
    forecast_type='load',  # 添加预测类型参数
    historical_days=8,  # 添加历史天数参数
    forecast_end_date=None  # 添加结束日期参数
):
    """
    执行日前预测，并在预测开始时应用平滑过渡
    
    Args:
        data_path: 数据路径
        forecast_date: 要预测的开始日期（字符串，格式：'YYYY-MM-DD'）
        peak_hours: 高峰时段起止小时，默认为(8, 20)
        valley_hours: 低谷时段起止小时，默认为(0, 7)
        peak_weight: 高峰时段权重，默认为2.5
        valley_weight: 低谷时段权重，默认为1.5
        dataset_id: 数据集标识，默认为"上海"
        max_allowed_diff_pct: 允许的最大百分比差异，默认5.0%
        forecast_type: 预测类型，可以是'load'、'pv'或'wind'
        historical_days: 使用的历史数据天数，默认为8天
        forecast_end_date: 要预测的结束日期（字符串，格式：'YYYY-MM-DD'），默认为None（单日预测）
    """
    # 确定日期范围描述
    if forecast_end_date is None:
        date_range_str = forecast_date
    else:
        date_range_str = f"{forecast_date}至{forecast_end_date}"
    
    print(f"=== 执行平滑过渡的日前{forecast_type}预测：{date_range_str} ===", flush=True)
    print(f"使用 {historical_days} 天的历史数据进行特征提取和预测", flush=True)
    
    # --- 更新: 使用包含 forecast_type 的路径 --- 
    model_base_name = 'convtrans_peak' 
    model_dir = f"models/{model_base_name}/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/{model_base_name}/{forecast_type}/{dataset_id}"
    results_dir = f"results/day_ahead/{model_base_name}/{forecast_type}/{dataset_id}" # <-- 使用 model_base_name
    # -------------------------------------------
    
    # 先执行基础的日前预测
    results_df = perform_day_ahead_forecast(
        data_path=data_path,
        forecast_date=forecast_date,
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=peak_weight,
        valley_weight=valley_weight,
        dataset_id=dataset_id,
        forecast_type=forecast_type,
        historical_days=historical_days,  # 传递历史天数参数
        forecast_end_date=forecast_end_date  # 传递结束日期参数
    )
    
    if results_df is None or results_df.empty:
        print("基础日前预测失败，无法进行平滑过渡。", flush=True)
        return None
        
    print("应用平滑过渡算法优化预测结果...", flush=True)
    
    # 对结果应用平滑处理，需要按日期分组处理
    if 'prediction_date' in results_df.columns:
        # 多天预测，按日期分组处理
        grouped_results = []
        for date, group in results_df.groupby('prediction_date'):
            # print(f"平滑处理 {date} 的预测结果...")
            smoothed_group = apply_daily_smoothing(group, max_allowed_diff_pct)
            grouped_results.append(smoothed_group)
        
        # 合并处理后的结果
        results_df = pd.concat(grouped_results, ignore_index=True)
    else:
        # 单天预测，直接处理
        results_df = apply_daily_smoothing(results_df, max_allowed_diff_pct)

    return results_df

# 抽取日常平滑函数，用于被两个平滑预测函数复用
def apply_daily_smoothing(day_results_df, max_allowed_diff_pct=5.0):
    """
    对单日预测结果应用平滑处理
    
    Args:
        day_results_df: 单日预测结果DataFrame
        max_allowed_diff_pct: 允许的最大百分比差异
    
    Returns:
        DataFrame: 平滑处理后的预测结果
    """
    # 创建结果副本以避免修改原始数据
    smoothed_df = day_results_df.copy()
    
    # 获取原始预测值
    original_predictions = smoothed_df['predicted'].values.copy()
    
    # 应用平滑算法
    for i in range(1, len(original_predictions)):
        current_pred = original_predictions[i]
        prev_pred = smoothed_df['predicted'].values[i-1]
        
        # 计算百分比差异
        if prev_pred > 0:
            pct_diff = abs(current_pred - prev_pred) / prev_pred * 100
        else:
            pct_diff = 100  # 如果前一个值为0，认为差异很大
        
        # 如果差异超过阈值，应用平滑
        if pct_diff > max_allowed_diff_pct:
            # 计算允许的最大差异值
            max_diff = prev_pred * max_allowed_diff_pct / 100
            
            # 根据差异方向调整当前值
            if current_pred > prev_pred:
                smoothed_df.at[i, 'predicted'] = prev_pred + max_diff
            else:
                smoothed_df.at[i, 'predicted'] = prev_pred - max_diff
    
    return smoothed_df

def perform_day_ahead_forecast_with_enhanced_smoothing(
    data_path,
    forecast_date,
    peak_hours=(8, 20),
    valley_hours=(0, 7),
    peak_weight=2.5,
    valley_weight=1.5,
    dataset_id='上海',
    max_allowed_diff_pct=5.0,  # 允许的最大百分比差异，可调整
    smoothing_window=24,       # 平滑窗口大小（点数）
    use_historical_patterns=True,  # 是否使用历史模式辅助调整
    historical_days=8,         # 使用历史几天的数据
    forecast_type='load',      # 预测类型参数
    forecast_end_date=None     # 添加结束日期参数
):
    """
    执行具有增强平滑功能的日前预测
    
    Args:
        data_path: 数据路径
        forecast_date: 要预测的开始日期（字符串，格式：'YYYY-MM-DD'）
        peak_hours: 高峰时段起止小时，默认为(8, 20)
        valley_hours: 低谷时段起止小时，默认为(0, 7)
        peak_weight: 高峰时段权重，默认为2.5
        valley_weight: 低谷时段权重，默认为1.5
        dataset_id: 数据集标识，默认为"上海"
        max_allowed_diff_pct: 允许的最大百分比差异，默认5.0%
        smoothing_window: 平滑窗口大小（点数），默认24
        use_historical_patterns: 是否使用历史模式辅助调整，默认True
        historical_days: 使用的历史数据天数，默认为8天
        forecast_type: 预测类型，可以是'load'、'pv'或'wind'
        forecast_end_date: 要预测的结束日期（字符串，格式：'YYYY-MM-DD'），默认为None（单日预测）
    """
    # 确定日期范围描述
    if forecast_end_date is None:
        date_range_str = forecast_date
    else:
        date_range_str = f"{forecast_date}至{forecast_end_date}"
    
    print(f"=== 执行增强平滑的日前{forecast_type}预测：{date_range_str} ===", flush=True)
    print(f"使用 {historical_days} 天的历史数据进行特征提取和预测", flush=True)
    
    # --- 更新: 使用包含 forecast_type 的路径 (虽然基础预测函数会处理，但保持一致性) ---
    model_base_name = 'convtrans_peak' 
    model_dir = f"models/{model_base_name}/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/{model_base_name}/{forecast_type}/{dataset_id}"
    results_dir = f"results/day_ahead/{model_base_name}/{forecast_type}/{dataset_id}" # <-- 使用 model_base_name
    # -------------------------------------------

    # 先执行基础的日前预测
    results_df = perform_day_ahead_forecast(
        data_path=data_path,
        forecast_date=forecast_date,
        peak_hours=peak_hours,
        valley_hours=valley_hours,
        peak_weight=peak_weight,
        valley_weight=valley_weight,
        dataset_id=dataset_id,
        forecast_type=forecast_type,
        historical_days=historical_days,  # 传递历史天数参数
        forecast_end_date=forecast_end_date  # 传递结束日期参数
    )

    if results_df is None or results_df.empty:
        print("基础日前预测失败，无法进行增强平滑。", flush=True)
        return None

    print("应用增强平滑算法优化预测结果...", flush=True)
    
    # 加载时间序列数据用于历史模式分析（所有日期共用）
    if use_historical_patterns:
        print(f"从 {data_path} 加载时间序列数据用于历史模式分析...", flush=True)
        ts_data = pd.read_csv(data_path, index_col=0)
        ts_data.index = pd.to_datetime(ts_data.index)
    
    # 对结果应用增强平滑处理，需要按日期分组处理
    if 'prediction_date' in results_df.columns:
        # 多天预测，按日期分组处理
        grouped_results = []
        for date, group in results_df.groupby('prediction_date'):
            # print(f"增强平滑处理 {date} 的预测结果...")
            # 解析当前日期
            current_date = pd.to_datetime(date)
            
            # 对当前日期应用增强平滑
            smoothed_group = apply_enhanced_smoothing(
                group, current_date, ts_data if use_historical_patterns else None, 
                forecast_type, smoothing_window, max_allowed_diff_pct,
                historical_days, peak_hours
            )
            
            grouped_results.append(smoothed_group)
        
        # 合并处理后的结果
        results_df = pd.concat(grouped_results, ignore_index=True)
    else:
        # 单天预测，直接处理
        current_date = pd.to_datetime(forecast_date)
        results_df = apply_enhanced_smoothing(
            results_df, current_date, ts_data if use_historical_patterns else None,
            forecast_type, smoothing_window, max_allowed_diff_pct,
            historical_days, peak_hours
        )
    
    # # 计算和输出整体性能指标（如果有实际值）
    # if 'actual' in results_df.columns and not results_df['actual'].isna().all():
    #     valid_df = results_df.dropna(subset=['actual'])
    #     if len(valid_df) > 0:
    #         metrics = calculate_metrics(valid_df['actual'].values, valid_df['predicted'].values)
            
    #         print("\n增强平滑日前预测性能指标:")
    #         for metric, value in metrics.items():
    #             print(f"{metric}: {value:.4f}")
            
    #         # 计算高峰和非高峰时段的指标
    #         peak_df = valid_df[valid_df['is_peak'] == 1]
    #         non_peak_df = valid_df[valid_df['is_peak'] == 0]
            
    #         if len(peak_df) > 0:
    #             peak_metrics = calculate_metrics(peak_df['actual'].values, peak_df['predicted'].values)
    #             print("\n高峰时段预测性能:")
    #             for metric, value in peak_metrics.items():
    #                 print(f"{metric}: {value:.4f}")
            
    #         if len(non_peak_df) > 0:
    #             non_peak_metrics = calculate_metrics(non_peak_df['actual'].values, non_peak_df['predicted'].values)
    #             print("\n非高峰时段预测性能:")
    #             for metric, value in non_peak_metrics.items():
    #                 print(f"{metric}: {value:.4f}")
    
    return results_df

# 增强平滑处理函数，用于对单日预测结果进行增强平滑处理
def apply_enhanced_smoothing(day_results_df, current_date, ts_data=None, forecast_type='load', 
                            smoothing_window=24, max_allowed_diff_pct=5.0, 
                            historical_days=8, peak_hours=(8, 20)):
    """
    对单日预测结果应用增强平滑处理
    
    Args:
        day_results_df: 单日预测结果DataFrame
        current_date: 当前预测日期 (datetime对象)
        ts_data: 时间序列数据，用于历史模式分析
        forecast_type: 预测类型
        smoothing_window: 平滑窗口大小
        max_allowed_diff_pct: 最大允许差异百分比
        historical_days: 历史数据天数
        peak_hours: 高峰时段
    
    Returns:
        DataFrame: 增强平滑处理后的预测结果
    """
    # 创建结果副本以避免修改原始数据
    smoothed_df = day_results_df.copy()
    
    # 1. 应用移动平均平滑
    original_predictions = smoothed_df['predicted'].values.copy()
    smoothed_predictions = np.zeros_like(original_predictions)
    
    for i in range(len(original_predictions)):
        # 确定当前窗口
        window_start = max(0, i - smoothing_window // 2)
        window_end = min(len(original_predictions), i + smoothing_window // 2 + 1)
        window = original_predictions[window_start:window_end]
        
        # 计算加权平均（当前点权重更高）
        weights = np.ones_like(window)
        middle_idx = i - window_start
        if middle_idx < len(weights):
            weights[middle_idx] = 3  # 当前点权重更高
        
        smoothed_predictions[i] = np.average(window, weights=weights)
    
    # 2. 如果有时间序列数据，根据历史模式进一步调整
    if ts_data is not None:
        # 计算历史同类型天的数据
        start_date = current_date - timedelta(days=historical_days)
        historical_data = ts_data[(ts_data.index >= start_date) & (ts_data.index < current_date)].copy()
        
        # 给历史数据添加时间特征
        historical_data['hour'] = historical_data.index.hour
        historical_data['dayofweek'] = historical_data.index.dayofweek
        
        # 确定预测日是工作日还是周末
        is_weekend = current_date.dayofweek >= 5
        day_type = 'weekend' if is_weekend else 'workday'
        
        print(f"预测日 {current_date.strftime('%Y-%m-%d')} 是{'周末' if is_weekend else '工作日'}", flush=True)
        
        # 过滤出与预测日同类型的历史日
        if day_type == 'weekend':
            similar_days = historical_data[historical_data['dayofweek'] >= 5]
        else:
            similar_days = historical_data[historical_data['dayofweek'] < 5]
        
        # 检查是否有足够的历史数据
        if len(similar_days) > 0:
            # 修复: 使用numpy.unique()而不是.unique()方法
            num_similar_days = len(np.unique(similar_days.index.date))
            print(f"找到 {num_similar_days} 个类似的历史日用于模式分析", flush=True)
            
            # 计算每小时的平均模式和标准差
            hourly_patterns = similar_days.groupby('hour')[forecast_type].agg(['mean', 'std'])
            
            # 对预测进行微调
            for i, dt in enumerate(smoothed_df['datetime']):
                hour = dt.hour
                
                if hour in hourly_patterns.index:
                    mean_value = hourly_patterns.loc[hour, 'mean']
                    std_value = hourly_patterns.loc[hour, 'std']
                    
                    # 当前预测值与历史模式的差异
                    current_pred = smoothed_predictions[i]
                    pattern_diff = abs(current_pred - mean_value)
                    
                    # 如果差异超过2倍标准差，向历史均值靠拢
                    if pattern_diff > 2 * std_value and std_value > 0:
                        # 计算调整系数（差异越大，调整越大）
                        adjustment = 0.5 * (mean_value - current_pred)
                        smoothed_predictions[i] = current_pred + adjustment
    
    # 3. 应用平滑过渡约束（限制相邻点的变化率）
    for i in range(1, len(smoothed_predictions)):
        current_pred = smoothed_predictions[i]
        prev_pred = smoothed_predictions[i-1]
        
        # 计算百分比差异
        if prev_pred > 0:
            pct_diff = abs(current_pred - prev_pred) / prev_pred * 100
        else:
            pct_diff = 100  # 如果前一个值为0，认为差异很大
        
        # 如果差异超过阈值，应用约束
        if pct_diff > max_allowed_diff_pct:
            # 计算允许的最大差异值
            max_diff = prev_pred * max_allowed_diff_pct / 100
            
            # 根据差异方向调整当前值
            if current_pred > prev_pred:
                smoothed_predictions[i] = prev_pred + max_diff
            else:
                smoothed_predictions[i] = prev_pred - max_diff
    
    # 更新预测结果
    smoothed_df['predicted'] = smoothed_predictions
    
    # 确保包含小时特征
    if 'hour' not in smoothed_df.columns:
        smoothed_df['hour'] = smoothed_df['datetime'].dt.hour
    
    # 确保包含高峰标记
    if 'is_peak' not in smoothed_df.columns:
        smoothed_df['is_peak'] = smoothed_df['hour'].apply(lambda x: peak_hours[0] <= x <= peak_hours[1]).astype(int)
    
    return smoothed_df

def perform_weather_aware_day_ahead_forecast(data_path, forecast_date, weather_features=None,
                                           peak_hours=(8, 20), valley_hours=(0, 6),
                                           peak_weight=2.5, valley_weight=1.5,
                                           dataset_id="福建", forecast_type='load',
                                           historical_days=8, forecast_end_date=None):
    """
    执行基于天气数据的日前负荷预测
    
    参数:
    data_path: 包含天气和负荷数据的CSV文件路径
    forecast_date: 预测开始日期 (YYYY-MM-DD格式)
    weather_features: 天气特征列表，None表示自动识别
    peak_hours: 高峰时段 (开始小时, 结束小时)
    valley_hours: 低谷时段 (开始小时, 结束小时)
    peak_weight: 高峰时段权重
    valley_weight: 低谷时段权重
    dataset_id: 数据集标识符
    forecast_type: 预测类型
    historical_days: 使用的历史天数
    forecast_end_date: 预测结束日期，None表示单日预测
    
    返回:
    dict: 包含预测结果和评估指标的字典
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from models.torch_models import WeatherAwareConvTransformer
    from data.dataset_builder import DatasetBuilder
    from utils.scaler_manager import ScalerManager
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print(f"\n=== 天气感知日前预测 ===")
    print(f"数据文件: {data_path}")
    print(f"预测日期范围: {forecast_date} 到 {forecast_end_date or forecast_date}")
    print(f"数据集: {dataset_id}")
    print(f"高峰时段: {peak_hours[0]}:00 - {peak_hours[1]}:00")
    print(f"低谷时段: {valley_hours[0]}:00 - {valley_hours[1]}:00")
    print(f"使用历史天数: {historical_days}")
    
    # 解析日期
    start_date = pd.to_datetime(forecast_date)
    
    if forecast_end_date:
        end_date = pd.to_datetime(forecast_end_date)
        if end_date < start_date:
            raise ValueError("结束日期不能早于开始日期")
    else:
        end_date = start_date
    
    # 设置路径
    model_dir = f"models/convtrans_weather/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/convtrans_weather/{forecast_type}/{dataset_id}"
    results_dir = f"results/weather_aware_day_ahead_test"
    
    # 确保目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查模型和标准化器是否存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"标准化器目录不存在: {scaler_dir}")
    
    # 加载训练好的模型
    print("加载天气感知模型...")
    model = WeatherAwareConvTransformer.load(save_dir=model_dir)
    
    # 加载标准化器
    print("加载标准化器...")
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    # 加载X和y标准化器
    scaler_manager.load_scaler('X')
    scaler_manager.load_scaler('y')
    
    # 读取天气和负荷数据
    print("读取数据...")
    df = pd.read_csv(data_path)
    
    # 确保datetime列存在并转换为datetime类型
    if 'datetime' not in df.columns:
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("数据中必须包含 'datetime' 或 'timestamp' 列")
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 设置datetime为索引
    df = df.set_index('datetime')
    
    # 自动识别天气特征（如果未指定）
    if weather_features is None:
        # 排除非天气特征列
        non_weather_cols = ['load', 'hour', 'day', 'month', 'weekday', 'is_weekend', 
                           'timestamp', 'datetime', 'PARTY_ID', 'is_peak', 'is_valley', 
                           'distance_to_peak', 'peak_magnitude']
        weather_features = [col for col in df.columns if col not in non_weather_cols]
        print(f"自动识别天气特征: {weather_features}")
    
    # 初始化数据集构建器 - 使用与训练时相同的配置
    dataset_builder = DatasetBuilder(seq_length=96, pred_horizon=1, standardize=False)
    
    # 初始化预测结果
    all_predictions = []
    all_actuals = []
    all_timestamps = []
    
    # 为光伏预测定义夜间时段
    if forecast_type == 'pv':
        # 光伏夜间时段：18:00-06:00（日落到日出）
        night_hours_start = 18
        night_hours_end = 6
        print(f"光伏预测模式：将在夜间时段({night_hours_start}:00-{night_hours_end}:00)应用零输出约束")
    
    # 逐日进行预测
    current_date = start_date
    
    while current_date <= end_date:
        print(f"\n预测日期: {current_date.strftime('%Y-%m-%d')}")
        
        # 计算需要的历史数据范围
        history_start = current_date - timedelta(days=historical_days)
        history_end = current_date - timedelta(minutes=15)  # 预测开始前的最后一个时间点
        
        # 提取历史数据
        try:
            historical_data = df.loc[history_start:history_end].copy()
        except KeyError:
            print(f"警告: 无法获取 {history_start} 到 {history_end} 的历史数据")
            current_date += timedelta(days=1)
            continue
        
        if len(historical_data) < 96:  # 至少需要96个时间点
            print(f"警告: 历史数据不足 ({len(historical_data)} < 96)，跳过此日期")
            current_date += timedelta(days=1)
            continue
        
        # 使用与训练时相同的特征工程方法
        print("应用特征工程...")
        # 根据预测类型使用正确的列名
        value_column = forecast_type  # 'load', 'pv', 或 'wind'
        enhanced_data = dataset_builder.build_dataset_with_peak_awareness(
            df=historical_data,
            date_column=None,  # 已经设置为索引
            value_column=value_column,
            peak_hours=peak_hours,
            valley_hours=valley_hours,
            peak_weight=peak_weight,
            valley_weight=valley_weight
        )
        
        # 准备预测当天的时间戳
        prediction_timestamps = pd.date_range(
            start=current_date,
            end=current_date + timedelta(days=1) - timedelta(minutes=15),
            freq='15T'
        )
        
        # 检查预测日期的天气数据是否可用
        forecast_weather_data = df.loc[prediction_timestamps[0]:prediction_timestamps[-1]]
        if len(forecast_weather_data) < len(prediction_timestamps):
            print(f"警告: 预测日期 {current_date.strftime('%Y-%m-%d')} 的天气数据不完整")
            current_date += timedelta(days=1)
            continue
        
        # 初始化当日预测结果
        daily_predictions = []
        daily_actuals = []
        
        # 递归滚动预测每个时间点
        print(f"开始递归预测，共 {len(prediction_timestamps)} 个时间点...")
        
        # 预先准备完整的预测期天气数据，确保特征一致性
        forecast_weather_enhanced = dataset_builder.build_dataset_with_peak_awareness(
            df=forecast_weather_data,
            date_column=None,
            value_column=value_column,  # 使用正确的列名
            peak_hours=peak_hours,
            valley_hours=valley_hours,
            peak_weight=peak_weight,
            valley_weight=valley_weight
        )
        
        for i, pred_timestamp in enumerate(prediction_timestamps):
            try:
                # 获取最新的历史数据（包括之前的预测值）
                if i == 0:
                    # 第一个预测点，使用原始历史数据
                    current_history = enhanced_data.copy()
                else:
                    # 后续预测点，需要更新历史数据
                    # 使用预处理好的天气特征，只更新负荷值
                    last_timestamp = prediction_timestamps[i-1]
                    last_prediction = daily_predictions[-1]
                    
                    # 从预处理的天气数据中获取对应时间点的特征
                    if last_timestamp in forecast_weather_enhanced.index:
                        new_row = forecast_weather_enhanced.loc[last_timestamp:last_timestamp].copy()
                        
                        # 更新目标值为预测值
                        new_row[value_column] = last_prediction
                        
                        # 重新计算依赖于目标值的特征
                        # 更新滞后特征（使用历史数据中的最新值）
                        if len(current_history) > 0:
                            # 获取最近的目标值用于滞后特征
                            recent_values = current_history[value_column].tail(168).values  # 最近一周的数据
                            
                            # 更新滞后特征（使用统一的列名前缀）
                            lag_prefix = f"{value_column}_lag"
                            if len(recent_values) >= 1:
                                new_row[f'{lag_prefix}_1'] = recent_values[-1]
                            if len(recent_values) >= 4:
                                new_row[f'{lag_prefix}_4'] = recent_values[-4]
                            if len(recent_values) >= 96:
                                new_row[f'{lag_prefix}_24'] = recent_values[-96]
                            if len(recent_values) >= 192:
                                new_row[f'{lag_prefix}_48'] = recent_values[-192]
                            if len(recent_values) >= 672:
                                new_row[f'{lag_prefix}_168'] = recent_values[-672]
                        
                        # 检查列是否匹配
                        if set(new_row.columns) != set(current_history.columns):
                            # 确保列顺序一致
                            new_row = new_row.reindex(columns=current_history.columns, fill_value=0)
                        
                        # 将新数据添加到历史数据中
                        current_history = pd.concat([current_history, new_row])
                        current_history = current_history.tail(len(enhanced_data))  # 保持相同长度
                    else:
                        # 使用最后一行数据作为模板
                        new_row = current_history.tail(1).copy()
                        new_row.index = [last_timestamp]
                        new_row[value_column] = last_prediction
                        
                        current_history = pd.concat([current_history, new_row])
                        current_history = current_history.tail(len(enhanced_data))
                
                # 验证数据完整性
                if len(current_history) < 96:
                    daily_predictions.append(0.0)
                    daily_actuals.append(np.nan)
                    continue
                
                # 检查是否有NaN值
                nan_count = current_history.isnull().sum().sum()
                if nan_count > 0:
                    # 填充NaN值
                    current_history = current_history.ffill().fillna(0)
                
                # 准备模型输入
                # 取最后96个时间点作为输入序列
                input_sequence = current_history.drop(columns=[value_column]).tail(96).values
                
                # 检查特征维度
                expected_features = input_sequence.shape[1]
                if i == 0:
                    print(f"模型期望特征数: {expected_features}")
                elif i < 5:
                    print(f"  时间点 {i} 特征数: {expected_features}")
                
                input_sequence = input_sequence.reshape(1, 96, -1)  # [1, seq_len, features]
                
                # 应用标准化
                batch_size, seq_len, n_features = input_sequence.shape
                input_reshaped = input_sequence.reshape(batch_size, seq_len * n_features)
                input_scaled = scaler_manager.transform('X', input_reshaped)
                input_scaled = input_scaled.reshape(batch_size, seq_len, n_features)
                
                # 进行预测
                prediction = model.predict(input_scaled)
                
                # 反标准化预测结果
                prediction_scaled = prediction.reshape(-1, 1)
                prediction_original = scaler_manager.inverse_transform('y', prediction_scaled)
                prediction_value = float(prediction_original[0, 0])
                
                # 添加合理性检查
                if np.isnan(prediction_value) or np.isinf(prediction_value):
                    if len(daily_predictions) > 0:
                        prediction_value = daily_predictions[-1]
                    else:
                        prediction_value = 30000.0 if forecast_type == 'load' else 0.0  # 根据预测类型使用合理的默认值
                elif prediction_value < 0:
                    if forecast_type == 'pv':
                        print(f"警告: 光伏预测值为负数 ({prediction_value})，设置为0")
                    prediction_value = 0.0
                
                # 光伏夜间约束处理
                if forecast_type == 'pv':
                    current_hour = pred_timestamp.hour
                    # 检查是否在夜间时段
                    is_night_time = (current_hour >= night_hours_start) or (current_hour <= night_hours_end)
                    
                    if is_night_time:
                        # 夜间光伏出力设置为0或接近0的小值
                        if prediction_value > 5.0:  # 如果预测值明显大于0，记录警告
                            if i < 5 or i % 48 == 0:  # 减少日志频率
                                print(f"  夜间光伏约束: {pred_timestamp.strftime('%H:%M')} 预测值从 {prediction_value:.2f} 调整为 0.0")
                        prediction_value = 0.0
                    else:
                        # 白天时段，确保预测值为非负
                        if prediction_value < 0:
                            prediction_value = 0.0
                
                daily_predictions.append(prediction_value)
                
                # 获取实际值（如果可用）
                if pred_timestamp in df.index:
                    actual_value = df.loc[pred_timestamp, value_column]
                    daily_actuals.append(actual_value)
                else:
                    daily_actuals.append(np.nan)
                
                if (i + 1) % 24 == 0:  # 每24个点（6小时）打印一次进度
                    print(f"  已完成 {i + 1}/{len(prediction_timestamps)} 个预测点，最新预测值: {prediction_value:.2f}")
                    
            except Exception as e:
                print(f"预测时间点 {pred_timestamp} 时出错: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈
                daily_predictions.append(0.0)
                daily_actuals.append(np.nan)
        
        # 添加到总结果中
        all_predictions.extend(daily_predictions)
        all_actuals.extend(daily_actuals)
        all_timestamps.extend(prediction_timestamps)
        
        print(f"完成日期 {current_date.strftime('%Y-%m-%d')} 的预测")
        current_date += timedelta(days=1)
    
    # 编译结果
    results_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'predicted': all_predictions,
        'actual': all_actuals
    })
    
    # 对于光伏预测，额外进行误差统计和修正
    if forecast_type == 'pv':
        # 统计夜间预测情况
        results_df['hour'] = pd.to_datetime(results_df['timestamp']).dt.hour
        night_mask = (results_df['hour'] >= night_hours_start) | (results_df['hour'] <= night_hours_end)
        
        night_predictions = results_df[night_mask]
        day_predictions = results_df[~night_mask]
        
        print(f"\n=== 光伏预测夜间约束统计 ===")
        print(f"夜间预测点数: {len(night_predictions)}")
        print(f"白天预测点数: {len(day_predictions)}")
        
        if len(night_predictions) > 0:
            night_nonzero = (night_predictions['predicted'] > 0.1).sum()
            print(f"夜间非零预测点数（应为0）: {night_nonzero}")
            
            # 如果仍有夜间非零预测，强制设置为0
            if night_nonzero > 0:
                print(f"强制将 {night_nonzero} 个夜间非零预测设置为0")
                results_df.loc[night_mask, 'predicted'] = 0.0
                all_predictions = results_df['predicted'].tolist()
    
    # 计算评估指标（改进误差计算，避免除零问题）
    valid_mask = ~np.isnan(all_actuals)
    if valid_mask.sum() > 0:
        valid_predictions = np.array(all_predictions)[valid_mask]
        valid_actuals = np.array(all_actuals)[valid_mask]
        
        mae = np.mean(np.abs(valid_predictions - valid_actuals))
        rmse = np.sqrt(np.mean((valid_predictions - valid_actuals) ** 2))
        
        # 改进MAPE计算，避免除零问题
        if forecast_type == 'pv':
            # 对于光伏预测，只计算实际值大于阈值的时段的MAPE
            significant_mask = valid_actuals > 10.0  # 只考虑实际值大于10MW的时段
            if significant_mask.sum() > 0:
                significant_predictions = valid_predictions[significant_mask]
                significant_actuals = valid_actuals[significant_mask]
                mape = np.mean(np.abs((significant_predictions - significant_actuals) / significant_actuals)) * 100
                print(f"MAPE计算基于{significant_mask.sum()}个有效时段（实际值>10MW）")
            else:
                mape = 0.0
                print("警告: 没有足够的有效时段计算MAPE")
        else:
            # 对于负荷预测，使用标准MAPE计算
            nonzero_mask = valid_actuals != 0
            if nonzero_mask.sum() > 0:
                mape = np.mean(np.abs((valid_predictions[nonzero_mask] - valid_actuals[nonzero_mask]) / valid_actuals[nonzero_mask])) * 100
            else:
                mape = 0.0
        
        print(f"\n=== 预测评估结果 ===")
        print(f"总预测点数: {len(all_predictions)}")
        print(f"有效评估点数: {valid_mask.sum()}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
    else:
        mae = rmse = mape = 0.0
        print(f"\n=== 预测完成 ===")
        print(f"总预测点数: {len(all_predictions)}")
        print(f"无实际值可用于评估")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"{results_dir}/weather_day_ahead_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"预测结果已保存到: {results_file}")
    
    # 生成可视化
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 8))
        plt.plot(results_df['timestamp'], results_df['predicted'], 
                label='预测值', linewidth=2, alpha=0.8)
        
        if valid_mask.sum() > 0:
            plt.plot(results_df['timestamp'], results_df['actual'], 
                    label='实际值', linewidth=2, alpha=0.8)
        
        plt.title(f'天气感知日前{forecast_type}预测结果 - {dataset_id}')
        plt.xlabel('时间')
        ylabel = '负荷 (MW)' if forecast_type == 'load' else ('光伏出力 (MW)' if forecast_type == 'pv' else '风电出力 (MW)')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_file = f"{results_dir}/weather_day_ahead_forecast_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"预测图表已保存到: {chart_file}")
        
    except Exception as e:
        print(f"生成图表时出错: {e}")
    
    return {
        'predictions': all_predictions,
        'actuals': all_actuals,
        'timestamps': all_timestamps,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'results_file': results_file
    }