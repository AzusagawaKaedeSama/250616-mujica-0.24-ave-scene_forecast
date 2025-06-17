import os
import time
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import sys
# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录 (scripts/forecast 目录)
current_script_dir = os.path.dirname(current_script_path)
# 获取 forecast 目录的上级目录 (scripts 目录)
scripts_dir = os.path.dirname(current_script_dir)
# 获取项目根目录 (scripts 目录的上级目录)
project_root_dir = os.path.dirname(scripts_dir)

# 将项目根目录添加到 Python 搜索路径中
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# 确保可以从正确的路径导入
from models.torch_models import PeakAwareConvTransformer # 假设这是基础模型
from utils.scaler_manager import ScalerManager
from data.dataset_builder import DatasetBuilder # 假设用于数据准备

logger = logging.getLogger(__name__)

class AdaptivePIDController:
    """自适应PID控制器"""
    def __init__(self, kp=0.7, ki=0.05, kd=0.1, dt=1.0, 
                 adaptation_rate=0.01, stability_threshold=0.05, 
                 error_history_size=100,
                 min_kp=0.1, max_kp=2.0,
                 min_ki=0.001, max_ki=0.5,
                 min_kd=0.01, max_kd=0.5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.initial_kp, self.initial_ki, self.initial_kd = kp, ki, kd # Store initial for reset or reference
        self.dt = dt
        self.integral = 0
        self.previous_error = 0
        
        self.adaptation_rate = adaptation_rate
        self.stability_threshold = stability_threshold # Percentage of typical value
        self.error_history = []
        self.output_history = [] # Stores PID correction values
        self.error_history_size = error_history_size

        # Parameter limits
        self.min_kp, self.max_kp = min_kp, max_kp
        self.min_ki, self.max_ki = min_ki, max_ki
        self.min_kd, self.max_kd = min_kd, max_kd
        
        self.typical_value_scale = 100 # Placeholder, should be updated with actual data scale

    def update_typical_value_scale(self, scale):
        """Update the typical scale of the value being controlled for threshold calculations."""
        if scale > 0:
            self.typical_value_scale = scale

    def compute(self, error, enable_correction=True):
        """计算PID输出"""
        self.error_history.append(error)
        if len(self.error_history) > self.error_history_size:
            self.error_history.pop(0)

        if not enable_correction: # 如果不启用修正（例如预训练阶段）
            self.integral += error * self.dt
            # Limit integral to prevent windup, scaled by typical value
            max_integral_abs = self.typical_value_scale * 0.5 # Example: integral shouldn't cause more than 50% correction alone
            self.integral = max(min(self.integral, max_integral_abs), -max_integral_abs)
            
            derivative = (error - self.previous_error) / self.dt if self.dt > 0 else 0
            self.previous_error = error
            return 0.0 # 不应用修正

        self.integral += error * self.dt
        max_integral_abs = self.typical_value_scale * 0.5 
        self.integral = max(min(self.integral, max_integral_abs), -max_integral_abs)
        
        derivative = (error - self.previous_error) / self.dt if self.dt > 0 else 0
        
        pid_output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        
        self.output_history.append(pid_output)
        if len(self.output_history) > self.error_history_size:
             self.output_history.pop(0)
            
        return pid_output

    def _adapt_parameters(self):
        """根据历史误差和输出自适应调整PID参数。"""
        if len(self.error_history) < self.error_history_size // 2: # 需要足够数据，例如一半的窗口
            # logger.debug(f"PID Adapt Skip: Not enough error history ({len(self.error_history)} < {self.error_history_size // 2})")
            return

        # Use a shorter recent window for adaptation decisions
        # Consider at least 20 points or 1/5th of history, whichever is larger, for adaptation decisions.
        # Ensure we don't try to take more than available. Max with 1 to avoid issues with tiny history_size.
        window_for_stats = max(1, min(len(self.error_history), max(20, int(self.error_history_size * 0.2)))) 
        recent_errors = np.array(self.error_history[-window_for_stats:])
        
        avg_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)
        
        # Dynamic stability threshold based on typical value scale
        current_stability_threshold_abs = self.stability_threshold * self.typical_value_scale
        # Log the inputs to adaptation decision
        logger.debug(f"PID Adapt Stats: error_history_len={len(self.error_history)}, recent_err_window={window_for_stats}, avg_error={avg_error:.4f}, std_error={std_error:.4f}, typical_scale={self.typical_value_scale:.2f}, stab_thresh_abs={current_stability_threshold_abs:.4f}")

        # --- Ki Adaptation (Integral Action) ---
        avg_error_check_threshold = current_stability_threshold_abs * 0.5
        logger.debug(f"PID Adapt Ki Check: abs(avg_error)={abs(avg_error):.4f} > threshold={avg_error_check_threshold:.4f} ? {abs(avg_error) > avg_error_check_threshold}")
        # If persistent offset (average error is high compared to threshold)
        if abs(avg_error) > avg_error_check_threshold: # e.g. > 2.5% of typical value
            ki_adjustment = self.adaptation_rate * (self.initial_ki * 0.1) # Adjust Ki proportionally to its initial value
            logger.debug(f"PID Adapt Ki Action: Modifying Ki. Initial_Ki={self.initial_ki:.4f}, Rate={self.adaptation_rate}, AdjustmentVal={ki_adjustment:.5f}, Integral={self.integral:.4f}, SignAvgErr={np.sign(avg_error)}")
            if np.sign(avg_error) == np.sign(self.integral) or self.integral == 0:
                self.ki += ki_adjustment * np.sign(avg_error)
                logger.debug(f"PID Adapt Ki: Increased Ki due to same sign or zero integral. New Ki before clip: {self.ki:.4f}")
            else:
                self.ki -= ki_adjustment * np.sign(avg_error)
                logger.debug(f"PID Adapt Ki: Decreased Ki due to opposite sign. New Ki before clip: {self.ki:.4f}")
            self.ki = np.clip(self.ki, self.min_ki, self.max_ki)
            logger.debug(f"PID Adapt Ki: Final Ki after clip: {self.ki:.4f}")

        # --- Kp and Kd Adaptation (Proportional and Derivative Action) ---
        std_error_check_threshold = current_stability_threshold_abs * 0.75
        logger.debug(f"PID Adapt KpKd Check (Oscillation): std_error={std_error:.4f} > threshold={std_error_check_threshold:.4f} ? {std_error > std_error_check_threshold}")
        # If high oscillation (std_error is high)
        if std_error > std_error_check_threshold: # e.g. > 3.75% of typical value
            kp_adj = self.adaptation_rate * (self.initial_kp * 0.05)
            kd_adj = self.adaptation_rate * (self.initial_kd * 0.05)
            logger.debug(f"PID Adapt KpKd Action (Oscillation): Modifying Kp/Kd. Kp_adj={kp_adj:.5f}, Kd_adj={kd_adj:.5f}")
            self.kp -= kp_adj
            self.kd += kd_adj
            logger.debug(f"PID Adapt KpKd (Oscillation): New Kp before clip: {self.kp:.3f}, New Kd before clip: {self.kd:.3f}")
        
        # Low error and low oscillation check
        avg_err_low_thresh = current_stability_threshold_abs * 0.1
        std_err_low_thresh = current_stability_threshold_abs * 0.2
        logger.debug(f"PID Adapt KpKd Check (Stable): abs(avg_error)={abs(avg_error):.4f} < low_avg_thresh={avg_err_low_thresh:.4f} ? {abs(avg_error) < avg_err_low_thresh}")
        logger.debug(f"PID Adapt KpKd Check (Stable): std_error={std_error:.4f} < low_std_thresh={std_err_low_thresh:.4f} ? {std_error < std_err_low_thresh}")
        # If very low error and low oscillation (system is stable and accurate)
        if abs(avg_error) < avg_err_low_thresh and std_error < std_err_low_thresh:
            kp_adj = self.adaptation_rate * (self.initial_kp * 0.01)
            kd_adj = self.adaptation_rate * (self.initial_kd * 0.01)
            logger.debug(f"PID Adapt KpKd Action (Stable): Reducing Kp/Kd. Kp_adj={kp_adj:.5f}, Kd_adj={kd_adj:.5f}")
            self.kp -= kp_adj
            self.kd -= kd_adj
            logger.debug(f"PID Adapt KpKd (Stable): New Kp before clip: {self.kp:.3f}, New Kd before clip: {self.kd:.3f}")

        # Sluggishness check (distinct from the above, so use elif if mutually exclusive, or if if can follow)
        avg_err_sluggish_thresh = current_stability_threshold_abs * 0.2
        std_err_sluggish_thresh = current_stability_threshold_abs * 0.5
        is_sluggish = abs(avg_error) > avg_err_sluggish_thresh and std_error < std_err_sluggish_thresh
        logger.debug(f"PID Adapt Kp Check (Sluggish): abs(avg_error)={abs(avg_error):.4f} > sluggish_avg_thresh={avg_err_sluggish_thresh:.4f} ? {abs(avg_error) > avg_err_sluggish_thresh}")
        logger.debug(f"PID Adapt Kp Check (Sluggish): std_error={std_error:.4f} < sluggish_std_thresh={std_err_sluggish_thresh:.4f} ? {std_error < std_err_sluggish_thresh}")
        logger.debug(f"PID Adapt Kp Check (Sluggish): Is Sluggish? {is_sluggish}")
        # If sluggish (error consistently exists but isn't oscillating wildly)
        # This condition should likely be an elif to avoid multiple Kp adjustments in one cycle if conditions overlap.
        # However, if the logic intends for multiple small adjustments, `if` is fine. Assuming mutually exclusive for now:
        if is_sluggish and not (std_error > std_error_check_threshold) and not (abs(avg_error) < avg_err_low_thresh and std_error < std_err_low_thresh):
            kp_adj = self.adaptation_rate * (self.initial_kp * 0.05)
            logger.debug(f"PID Adapt Kp Action (Sluggish): Increasing Kp. Kp_adj={kp_adj:.5f}")
            self.kp += kp_adj
            logger.debug(f"PID Adapt Kp (Sluggish): New Kp before clip: {self.kp:.3f}")

        self.kp = np.clip(self.kp, self.min_kp, self.max_kp)
        self.kd = np.clip(self.kd, self.min_kd, self.max_kd)

        logger.debug(f"PID Adapt Final Params: Kp={self.kp:.3f}, Ki={self.ki:.4f}, Kd={self.kd:.3f}")


class SlidingWindowErrorAnalyzer:
    """滑动窗口误差分析器"""
    def __init__(self, window_size_hours=72):
        self.window_size_hours = window_size_hours 
        self.hourly_errors = {h: [] for h in range(24)} 
        self.hourly_correction_factors = {h: 1.0 for h in range(24)}
        self.forecast_interval_minutes = 15 

    def set_forecast_interval(self, interval_minutes):
        if interval_minutes > 0:
            self.forecast_interval_minutes = interval_minutes
        else:
            logger.warning("Forecast interval must be positive. Using default 15 minutes.")
            self.forecast_interval_minutes = 15


    def update(self, timestamp, actual, predicted, hour):
        """更新指定小时的误差数据"""
        if predicted is not None and actual is not None and predicted != 0:
            relative_error = (actual / predicted) - 1.0 
            self.hourly_errors[hour].append(relative_error)
            
            window_size_points = int(self.window_size_hours * (60 / self.forecast_interval_minutes))
            if window_size_points <=0: window_size_points = 1 

            if len(self.hourly_errors[hour]) > window_size_points: 
                self.hourly_errors[hour].pop(0)
            
            if self.hourly_errors[hour]:
                avg_relative_error = np.mean(self.hourly_errors[hour])
                self.hourly_correction_factors[hour] = avg_relative_error + 1.0
                self.hourly_correction_factors[hour] = max(0.8, min(self.hourly_correction_factors[hour], 1.2))


    def get_hourly_correction_factor(self, hour):
        """获取指定小时的修正因子"""
        return self.hourly_correction_factors.get(hour, 1.0)

    def get_pattern_statistics(self):
        """获取各时段误差模式统计（可选）"""
        stats = {}
        for hour, errors in self.hourly_errors.items():
            if errors:
                stats[hour] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'count': len(errors),
                    'correction_factor': self.hourly_correction_factors[hour]
                }
        return stats


def perform_rolling_forecast_with_pid_correction(
    data_path, 
    start_date, 
    end_date, 
    forecast_interval=15,
    peak_hours=(8, 20), 
    valley_hours=(0, 6),
    peak_weight=5.0, 
    valley_weight=1.5,
    apply_smoothing=False, 
    dataset_id='上海',
    forecast_type='load', 
    historical_days=8,
    pretrain_days=3,
    window_size_hours=72, 
    enable_adaptation=True,
    initial_kp=None,
    initial_ki=None,
    initial_kd=None,
    **kwargs 
):
    """
    执行带有PID误差修正的滚动预测。PID参数通过内部自适应调整。
    预训练阶段在实际预测开始前进行。
    """
    logger.info(f"开始执行带PID修正的滚动预测: {dataset_id} ({forecast_type})")
    logger.info(f"请求的预测时段: {start_date} to {end_date}, 间隔: {forecast_interval}min")
    logger.info(f"PID参数: 预训练天数={pretrain_days}, 窗口(小时)={window_size_hours}, 自适应={enable_adaptation}")
    if initial_kp is not None: logger.info(f"使用用户提供的初始PID: Kp={initial_kp}, Ki={initial_ki}, Kd={initial_kd}")

    model_base_name = PeakAwareConvTransformer.model_type 
    model_dir = f"models/{model_base_name}/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/{model_base_name}/{forecast_type}/{dataset_id}"
    results_dir = f"results/pid_corrected/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir)):
        raise FileNotFoundError(f"基础模型目录不存在或无模型: {model_dir}")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"基础模型缩放器目录不存在: {scaler_dir}")

    base_model = PeakAwareConvTransformer.load(save_dir=model_dir)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    seq_length = base_model.config.get('seq_length', int(1440 / forecast_interval))
    
    ts_data = pd.read_csv(data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    value_column = forecast_type 
    
    points_per_day = 24 * (60 / forecast_interval)
    desired_error_history_size = int(pretrain_days * points_per_day) 
    min_history_points = 50 
    max_history_points_by_days = int(7 * points_per_day) 
    max_history_points_cap = 1000 

    error_history_size_for_pid = max(min_history_points, desired_error_history_size)
    error_history_size_for_pid = min(error_history_size_for_pid, max_history_points_by_days, max_history_points_cap)
    logger.info(f"为PID控制器设置的 error_history_size: {error_history_size_for_pid} (基于 pretrain_days={pretrain_days}, interval={forecast_interval}min)")
    
    # Use provided initial PID params or defaults from AdaptivePIDController class
    pid_kwargs = {'dt': 1.0, 'adaptation_rate': 0.01 if enable_adaptation else 0, 'error_history_size': error_history_size_for_pid}
    if initial_kp is not None: pid_kwargs['kp'] = initial_kp
    if initial_ki is not None: pid_kwargs['ki'] = initial_ki
    if initial_kd is not None: pid_kwargs['kd'] = initial_kd
    pid_controller = AdaptivePIDController(**pid_kwargs)
    
    sample_data_for_scale = ts_data[value_column].dropna().iloc[:min(1000, len(ts_data[value_column].dropna()))]
    if not sample_data_for_scale.empty:
        pid_controller.update_typical_value_scale(sample_data_for_scale.mean())

    dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1)
    
    # --- Pre-training Phase ---
    actual_start_datetime = pd.to_datetime(start_date) # User requested prediction start
    
    if pretrain_days > 0:
        pretrain_loop_end_time = actual_start_datetime - timedelta(minutes=forecast_interval)
        pretrain_loop_start_time = actual_start_datetime - timedelta(days=pretrain_days)
        
        logger.info(f"开始PID预训练阶段: {pretrain_loop_start_time} to {pretrain_loop_end_time}")
        
        # Ensure enough data for pretraining
        min_data_needed_for_pretrain_start = pretrain_loop_start_time - timedelta(days=historical_days) - timedelta(minutes=seq_length*forecast_interval)
        if ts_data.index.min() > min_data_needed_for_pretrain_start:
            logger.warning(f"预训练数据不足，最早可用数据: {ts_data.index.min()}, 预训练需要开始于: {min_data_needed_for_pretrain_start}. 可能影响PID预训练效果。")
            # Adjust pretrain_loop_start_time if not enough history, or skip pretrain
            # For now, we'll proceed but PID might not learn well.
            # A more robust approach would be to adjust pretrain_loop_start_time or skip.

        current_pretrain_time = pretrain_loop_start_time
        while current_pretrain_time <= pretrain_loop_end_time:
            current_hour = current_pretrain_time.hour
            try:
                df_for_builder_end_pt = current_pretrain_time - timedelta(minutes=forecast_interval)
                df_for_builder_start_pt = df_for_builder_end_pt - timedelta(days=max(historical_days + 2, 10))
                df_for_builder_start_pt = max(df_for_builder_start_pt, ts_data.index.min())
                df_for_builder_pt = ts_data.loc[df_for_builder_start_pt:df_for_builder_end_pt].copy()

                if df_for_builder_pt.empty or len(df_for_builder_pt) < seq_length:
                    logger.debug(f"[Pretrain] 数据不足或为空 at {current_pretrain_time} for builder. Skipping.")
                    current_pretrain_time += timedelta(minutes=forecast_interval)
                    continue

                enhanced_data_pt = dataset_builder.build_dataset_with_peak_awareness(
                    df=df_for_builder_pt, date_column=None, value_column=value_column,
                    interval=forecast_interval, peak_hours=peak_hours, valley_hours=valley_hours,
                    peak_weight=peak_weight, valley_weight=valley_weight,
                    start_date=df_for_builder_pt.index.min().strftime('%Y-%m-%d %H:%M:%S'), 
                    end_date=df_for_builder_pt.index.max().strftime('%Y-%m-%d %H:%M:%S')
                )

                if len(enhanced_data_pt) < seq_length:
                    logger.debug(f"[Pretrain] 增强数据点数不足 at {current_pretrain_time} ({len(enhanced_data_pt)} < {seq_length}). Skipping.")
                    current_pretrain_time += timedelta(minutes=forecast_interval)
                    continue
                
                X_features_pt = enhanced_data_pt.iloc[-seq_length:].drop(columns=[value_column]).values
                X_features_reshaped_pt = X_features_pt.reshape(1, seq_length, -1)
                X_scaled_pt = scaler_manager.transform('X', X_features_reshaped_pt.reshape(1, -1)).reshape(X_features_reshaped_pt.shape)

                base_model_pred_pt_scaled = base_model.predict(X_scaled_pt)
                base_model_pred_pt = scaler_manager.inverse_transform('y', base_model_pred_pt_scaled).flatten()[0]
                
                actual_value_pt = ts_data.loc[current_pretrain_time, value_column] if current_pretrain_time in ts_data.index else np.nan

                if not np.isnan(base_model_pred_pt) and not np.isnan(actual_value_pt):
                    error_pt = actual_value_pt - base_model_pred_pt
                    pid_controller.compute(error_pt, enable_correction=False) # Update PID state, no correction output
                    if enable_adaptation:
                        pid_controller._adapt_parameters() # Adapt PID params during pretraining
                
                if current_pretrain_time.minute == 0 and current_pretrain_time.hour % 3 == 0:
                    logger.info(f"[Pretrain] Time: {current_pretrain_time}, Adapted PID: Kp={pid_controller.kp:.3f}, Ki={pid_controller.ki:.4f}, Kd={pid_controller.kd:.3f}")

            except Exception as e_pt:
                logger.error(f"[Pretrain] Error at {current_pretrain_time}: {e_pt}")
            
            current_pretrain_time += timedelta(minutes=forecast_interval)
        logger.info("PID预训练阶段完成。")

    # --- Main Forecasting Phase ---
    logger.info(f"开始主预测阶段: {actual_start_datetime} to {pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)}")
    all_results = []
    # pretrain_end_datetime = actual_start_datetime + timedelta(days=pretrain_days) # This definition is now for the old logic

    current_time = actual_start_datetime # Start main forecast from the user-requested start_date
    main_loop_end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)

    while current_time <= main_loop_end_datetime:
        # is_in_pretrain = current_time < pretrain_end_datetime # Old logic, pretrain is now separate
        current_hour = current_time.hour
        
        try:
            df_for_builder_end = current_time - timedelta(minutes=forecast_interval)
            df_for_builder_start = df_for_builder_end - timedelta(days=max(historical_days + 2, 10)) 
            df_for_builder_start = max(df_for_builder_start, ts_data.index.min())
            df_for_builder = ts_data.loc[df_for_builder_start:df_for_builder_end].copy()

            if df_for_builder.empty or len(df_for_builder) < seq_length :
                logger.warning(f"[MainForecast] 在 {current_time} 时，用于 DatasetBuilder 的数据为空或过短。跳过此点。")
                all_results.append({
                    'datetime': current_time, 'actual': ts_data.loc[current_time, value_column] if current_time in ts_data.index else np.nan, 
                    'base_predicted': np.nan, 'pid_correction': 0.0, 'predicted': np.nan, 
                    'is_peak': peak_hours[0] <= current_hour < peak_hours[1],
                    'kp': pid_controller.kp, 'ki': pid_controller.ki, 'kd': pid_controller.kd
                })
                current_time += timedelta(minutes=forecast_interval)
                continue
            
            enhanced_data_for_seq = dataset_builder.build_dataset_with_peak_awareness(
                df=df_for_builder, date_column=None, value_column=value_column,
                interval=forecast_interval, peak_hours=peak_hours, valley_hours=valley_hours,
                peak_weight=peak_weight, valley_weight=valley_weight,
                start_date=df_for_builder.index.min().strftime('%Y-%m-%d %H:%M:%S'), 
                end_date=df_for_builder.index.max().strftime('%Y-%m-%d %H:%M:%S')
            )

            if len(enhanced_data_for_seq) < seq_length:
                logger.warning(f"[MainForecast] 为 {current_time} 准备数据时，增强数据点数不足 ({len(enhanced_data_for_seq)} < {seq_length})。跳过。")
                all_results.append({
                    'datetime': current_time, 'actual': ts_data.loc[current_time, value_column] if current_time in ts_data.index else np.nan, 
                    'base_predicted': np.nan, 'pid_correction': 0.0, 'predicted': np.nan, 
                    'is_peak': peak_hours[0] <= current_hour < peak_hours[1],
                    'kp': pid_controller.kp, 'ki': pid_controller.ki, 'kd': pid_controller.kd
                })
                current_time += timedelta(minutes=forecast_interval)
                continue
            
            X_features = enhanced_data_for_seq.iloc[-seq_length:].drop(columns=[value_column]).values
            X_features_reshaped = X_features.reshape(1, seq_length, -1)
            X_scaled = scaler_manager.transform('X', X_features_reshaped.reshape(1, -1)).reshape(X_features_reshaped.shape)

            base_model_prediction_scaled = base_model.predict(X_scaled)
            base_model_prediction = scaler_manager.inverse_transform('y', base_model_prediction_scaled).flatten()[0]

        except Exception as model_pred_err:
            logger.error(f"[MainForecast] 在 {current_time} 进行基础模型预测时出错: {model_pred_err}")
            base_model_prediction = np.nan 

        actual_value = ts_data.loc[current_time, value_column] if current_time in ts_data.index else np.nan
        
        pid_correction = 0.0
        final_prediction = base_model_prediction

        if not np.isnan(base_model_prediction) and not np.isnan(actual_value):
            error = actual_value - base_model_prediction
            # error_analyzer.update(current_time, actual_value, base_model_prediction, current_hour) # SlidingWindowErrorAnalyzer can be used if needed
            
            pid_correction = pid_controller.compute(error, enable_correction=True) # Enable correction in main phase
            
            final_prediction = base_model_prediction + pid_correction
            final_prediction = max(0, final_prediction) 
        if enable_adaptation:
            pid_controller._adapt_parameters() 

        all_results.append({
            'datetime': current_time, 'actual': actual_value, 'base_predicted': base_model_prediction,
            'pid_correction': pid_correction,
            'predicted': final_prediction, 
            'is_peak': peak_hours[0] <= current_hour < peak_hours[1],
            'kp': pid_controller.kp, 'ki': pid_controller.ki, 'kd': pid_controller.kd
        })
        
        if current_time.minute == 0 and current_time.hour % 3 == 0 : 
             logger.info(f"[MainForecast] Time: {current_time}, Adapted PID: Kp={pid_controller.kp:.3f}, Ki={pid_controller.ki:.4f}, Kd={pid_controller.kd:.3f}")
        
        current_time += timedelta(minutes=forecast_interval)

    results_df = pd.DataFrame(all_results)
    
    # Ensure 'predicted' column exists even if all were NaN, for consistency
    if 'predicted' not in results_df.columns and not results_df.empty:
        results_df['predicted'] = np.nan
    elif results_df.empty : # Handle case where all_results was empty
        logger.warning("最终结果为空，无法生成有效的CSV。")
        # Create an empty dataframe with expected columns to avoid downstream errors if needed
        results_df = pd.DataFrame(columns=['datetime', 'actual', 'base_predicted', 'pid_correction', 'predicted', 'is_peak', 'kp', 'ki', 'kd'])


    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use actual_start_datetime for filename consistency with user's requested period
    filename = f"pid_corrected_forecast_{actual_start_datetime.strftime('%Y-%m-%d')}_{end_date}_{timestamp_str}.csv"
    results_df.to_csv(os.path.join(results_dir, filename), index=False)
    logger.info(f"PID修正预测结果已保存至 {os.path.join(results_dir, filename)}")

    return results_df


def advanced_error_correction_system(
    data_path, 
    start_date, 
    end_date,
    forecast_interval=15,
    dataset_id='上海',
    forecast_type='load', 
    peak_hours=(8, 20),
    valley_hours=(0, 6),
    peak_weight=5.0,
    valley_weight=1.5,
    apply_smoothing=False,
    historical_days=8,
    pretrain_days=3,
    window_size_hours=72,
    enable_pid=True,      
    enable_adaptation=True, 
    enable_pattern_learning=False, 
    enable_ensemble=False, # TODO: Implement ensemble logic if True
    initial_kp=None, # New parameter
    initial_ki=None, # New parameter
    initial_kd=None, # New parameter
    **kwargs 
):
    """
    高级误差修正系统，结合了PID控制、模式学习和可选的集成方法。
    预训练阶段在实际预测开始前进行。
    """
    logger.info(f"启动高级误差修正系统: {dataset_id} ({forecast_type})")
    logger.info(f"预测时段: {start_date} to {end_date}, 间隔: {forecast_interval}min")
    logger.info(f"参数: PID修正={enable_pid}, 自适应={enable_adaptation}, 模式学习={enable_pattern_learning}, 集成={enable_ensemble}")
    logger.info(f"PID参数: 预训练天数={pretrain_days}, 窗口(小时)={window_size_hours}")
    if initial_kp is not None: logger.info(f"使用用户提供的初始PID: Kp={initial_kp}, Ki={initial_ki}, Kd={initial_kd}")

    model_base_name = PeakAwareConvTransformer.model_type
    model_dir = f"models/{model_base_name}/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/{model_base_name}/{forecast_type}/{dataset_id}"
    results_dir = f"results/advanced_corrected/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir)):
        raise FileNotFoundError(f"基础模型目录不存在或无模型: {model_dir}")
    if not os.path.exists(scaler_dir):
        raise FileNotFoundError(f"基础模型缩放器目录不存在: {scaler_dir}")

    base_model = PeakAwareConvTransformer.load(save_dir=model_dir)
    scaler_manager = ScalerManager(scaler_path=scaler_dir)
    seq_length = base_model.config.get('seq_length', int(1440 / forecast_interval))
    
    ts_data = pd.read_csv(data_path, index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    value_column = forecast_type

    pid_controller = None
    if enable_pid:
        points_per_day = 24 * (60 / forecast_interval)
        desired_error_history_size = int(pretrain_days * points_per_day)
        min_history_points = 50
        max_history_points_by_days = int(7 * points_per_day)
        max_history_points_cap = 1000
        error_history_size_for_pid = max(min_history_points, desired_error_history_size)
        error_history_size_for_pid = min(error_history_size_for_pid, max_history_points_by_days, max_history_points_cap)
        logger.info(f"[Advanced] 为PID控制器设置的 error_history_size: {error_history_size_for_pid}")

        pid_kwargs_adv = {'dt': 1.0, 'adaptation_rate': 0.01 if enable_adaptation else 0, 'error_history_size': error_history_size_for_pid}
        if initial_kp is not None: pid_kwargs_adv['kp'] = initial_kp
        if initial_ki is not None: pid_kwargs_adv['ki'] = initial_ki
        if initial_kd is not None: pid_kwargs_adv['kd'] = initial_kd
        pid_controller = AdaptivePIDController(**pid_kwargs_adv)
        
        sample_data_for_scale = ts_data[value_column].dropna().iloc[:min(1000, len(ts_data[value_column].dropna()))]
        if not sample_data_for_scale.empty and pid_controller:
            pid_controller.update_typical_value_scale(sample_data_for_scale.mean())

    error_analyzer = None
    if enable_pattern_learning:
        error_analyzer = SlidingWindowErrorAnalyzer(window_size_hours=window_size_hours)
        error_analyzer.set_forecast_interval(forecast_interval)

    dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1)
    all_results = []

    actual_start_datetime = pd.to_datetime(start_date)

    # --- Pre-training Phase (for PID) ---
    if enable_pid and pid_controller and pretrain_days > 0:
        pretrain_loop_end_time = actual_start_datetime - timedelta(minutes=forecast_interval)
        pretrain_loop_start_time = actual_start_datetime - timedelta(days=pretrain_days)
        logger.info(f"[Advanced] 开始PID预训练阶段: {pretrain_loop_start_time} to {pretrain_loop_end_time}")

        min_data_needed_for_pretrain_start = pretrain_loop_start_time - timedelta(days=historical_days) - timedelta(minutes=seq_length*forecast_interval)
        if ts_data.index.min() > min_data_needed_for_pretrain_start:
            logger.warning(f"[Advanced-Pretrain] 预训练数据不足. 最早可用: {ts_data.index.min()}, 需要开始于: {min_data_needed_for_pretrain_start}.")

        current_pretrain_time = pretrain_loop_start_time
        while current_pretrain_time <= pretrain_loop_end_time:
            try:
                df_for_builder_end_pt = current_pretrain_time - timedelta(minutes=forecast_interval)
                df_for_builder_start_pt = df_for_builder_end_pt - timedelta(days=max(historical_days + 2, 10))
                df_for_builder_start_pt = max(df_for_builder_start_pt, ts_data.index.min())
                df_for_builder_pt = ts_data.loc[df_for_builder_start_pt:df_for_builder_end_pt].copy()

                if df_for_builder_pt.empty or len(df_for_builder_pt) < seq_length:
                    current_pretrain_time += timedelta(minutes=forecast_interval)
                    continue

                enhanced_data_pt = dataset_builder.build_dataset_with_peak_awareness(
                    df=df_for_builder_pt, date_column=None, value_column=value_column,
                    interval=forecast_interval, peak_hours=peak_hours, valley_hours=valley_hours,
                    peak_weight=peak_weight, valley_weight=valley_weight,
                    start_date=df_for_builder_pt.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    end_date=df_for_builder_pt.index.max().strftime('%Y-%m-%d %H:%M:%S')
                )
                if len(enhanced_data_pt) < seq_length:
                    current_pretrain_time += timedelta(minutes=forecast_interval)
                    continue
                
                X_features_pt = enhanced_data_pt.iloc[-seq_length:].drop(columns=[value_column]).values
                X_scaled_pt = scaler_manager.transform('X', X_features_pt.reshape(1, -1)).reshape(1, seq_length, -1)
                base_model_pred_pt_scaled = base_model.predict(X_scaled_pt)
                base_model_pred_pt = scaler_manager.inverse_transform('y', base_model_pred_pt_scaled).flatten()[0]
                
                actual_value_pt = ts_data.loc[current_pretrain_time, value_column] if current_pretrain_time in ts_data.index else np.nan

                if not np.isnan(base_model_pred_pt) and not np.isnan(actual_value_pt) and pid_controller:
                    error_pt = actual_value_pt - base_model_pred_pt
                    pid_controller.compute(error_pt, enable_correction=False)
                    if enable_adaptation:
                        pid_controller._adapt_parameters()
                
                if current_pretrain_time.minute == 0 and current_pretrain_time.hour % 3 == 0 and pid_controller:
                    logger.info(f"[Advanced-Pretrain] Time: {current_pretrain_time}, Adapted PID: Kp={pid_controller.kp:.3f}, Ki={pid_controller.ki:.4f}, Kd={pid_controller.kd:.3f}")
            except Exception as e_pt:
                logger.error(f"[Advanced-Pretrain] Error at {current_pretrain_time}: {e_pt}")
            current_pretrain_time += timedelta(minutes=forecast_interval)
        logger.info("[Advanced] PID预训练阶段完成。")

    # --- Main Forecasting Phase ---
    logger.info(f"[Advanced] 开始主预测阶段: {actual_start_datetime} to {pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)}")
    current_time = actual_start_datetime
    main_loop_end_datetime = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)

    while current_time <= main_loop_end_datetime:
        current_hour = current_time.hour
        base_model_prediction = np.nan
        pid_correction_val = 0.0
        pattern_correction_val = 0.0
        final_prediction = np.nan
        kp_val, ki_val, kd_val = (pid_controller.kp if pid_controller else np.nan, 
                                  pid_controller.ki if pid_controller else np.nan, 
                                  pid_controller.kd if pid_controller else np.nan)

        try:
            df_for_builder_end = current_time - timedelta(minutes=forecast_interval)
            df_for_builder_start = df_for_builder_end - timedelta(days=max(historical_days + 2, 10))
            df_for_builder_start = max(df_for_builder_start, ts_data.index.min())
            df_for_builder = ts_data.loc[df_for_builder_start:df_for_builder_end].copy()

            if df_for_builder.empty or len(df_for_builder) < seq_length:
                logger.warning(f"[Advanced-Main] 在 {current_time} 时数据不足。跳过。")
                # Append NaN or placeholder result
                all_results.append({
                    'datetime': current_time, 'actual': ts_data.loc[current_time, value_column] if current_time in ts_data.index else np.nan,
                    'base_predicted': np.nan, 'pid_correction': np.nan, 'pattern_correction': np.nan, 'predicted': np.nan,
                    'is_peak': peak_hours[0] <= current_hour < peak_hours[1],
                    'kp': kp_val, 'ki': ki_val, 'kd': kd_val
                })
                current_time += timedelta(minutes=forecast_interval)
                continue

            enhanced_data_for_seq = dataset_builder.build_dataset_with_peak_awareness(
                df=df_for_builder, date_column=None, value_column=value_column,
                interval=forecast_interval, peak_hours=peak_hours, valley_hours=valley_hours,
                peak_weight=peak_weight, valley_weight=valley_weight,
                start_date=df_for_builder.index.min().strftime('%Y-%m-%d %H:%M:%S'), 
                end_date=df_for_builder.index.max().strftime('%Y-%m-%d %H:%M:%S')
            )
            if len(enhanced_data_for_seq) < seq_length:
                logger.warning(f"[Advanced-Main] 为 {current_time} 增强数据点数不足。跳过。")
                all_results.append({
                    'datetime': current_time, 'actual': ts_data.loc[current_time, value_column] if current_time in ts_data.index else np.nan,
                    'base_predicted': np.nan, 'pid_correction': np.nan, 'pattern_correction': np.nan, 'predicted': np.nan,
                    'is_peak': peak_hours[0] <= current_hour < peak_hours[1],
                    'kp': kp_val, 'ki': ki_val, 'kd': kd_val
                })
                current_time += timedelta(minutes=forecast_interval)
                continue
            
            X_features = enhanced_data_for_seq.iloc[-seq_length:].drop(columns=[value_column]).values
            X_scaled = scaler_manager.transform('X', X_features.reshape(1, -1)).reshape(1, seq_length, -1)
            base_model_prediction_scaled = base_model.predict(X_scaled)
            base_model_prediction = scaler_manager.inverse_transform('y', base_model_prediction_scaled).flatten()[0]

        except Exception as model_pred_err:
            logger.error(f"[Advanced-Main] 在 {current_time} 基础模型预测出错: {model_pred_err}")
            # Fall through to append with NaN base_model_prediction

        actual_value = ts_data.loc[current_time, value_column] if current_time in ts_data.index else np.nan
        
        if not np.isnan(base_model_prediction) and not np.isnan(actual_value):
            error = actual_value - base_model_prediction
            
            if enable_pid and pid_controller:
                pid_correction_val = pid_controller.compute(error, enable_correction=True)
                if enable_adaptation:
                    pid_controller._adapt_parameters()
                kp_val, ki_val, kd_val = pid_controller.kp, pid_controller.ki, pid_controller.kd
            
            if enable_pattern_learning and error_analyzer:
                error_analyzer.update(current_time, actual_value, base_model_prediction, current_hour)
                pattern_correction_val = error_analyzer.get_hourly_correction_factor(current_hour)
            
            final_prediction = base_model_prediction + pid_correction_val + pattern_correction_val 
            final_prediction = max(0, final_prediction)
        else: # Case where base_model_prediction or actual_value is NaN
            final_prediction = base_model_prediction # Keep it NaN if base was NaN

        all_results.append({
            'datetime': current_time, 'actual': actual_value, 'base_predicted': base_model_prediction,
            'pid_correction': pid_correction_val, 'pattern_correction': pattern_correction_val,
            'predicted': final_prediction, 
            'is_peak': peak_hours[0] <= current_hour < peak_hours[1],
            'kp': kp_val, 'ki': ki_val, 'kd': kd_val
        })

        if current_time.minute == 0 and current_time.hour % 3 == 0 and pid_controller and enable_pid:
            logger.info(f"[Advanced-Main] Time: {current_time}, Adapted PID: Kp={pid_controller.kp:.3f}, Ki={pid_controller.ki:.4f}, Kd={pid_controller.kd:.3f}")

        current_time += timedelta(minutes=forecast_interval)

    results_df = pd.DataFrame(all_results)
    if 'predicted' not in results_df.columns and not results_df.empty:
        results_df['predicted'] = np.nan
    elif results_df.empty:
        logger.warning("[Advanced] 最终结果为空，无法生成CSV。")
        results_df = pd.DataFrame(columns=['datetime', 'actual', 'base_predicted', 'pid_correction', 'pattern_correction', 'predicted', 'is_peak', 'kp', 'ki', 'kd'])

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"advanced_corrected_forecast_{actual_start_datetime.strftime('%Y-%m-%d')}_{end_date}_{timestamp_str}.csv"
    results_df.to_csv(os.path.join(results_dir, filename), index=False)
    logger.info(f"[Advanced] 高级误差修正预测结果已保存至 {os.path.join(results_dir, filename)}")

    return results_df