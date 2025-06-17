import pandas as pd
import numpy as np
import json
import argparse
import os
from typing import Dict, Any

# 导入三类预测主函数（假设路径和函数名如下，如有不同请调整）
from scripts.forecast.day_ahead_forecast import perform_day_ahead_forecast
from scripts.forecast.forecast import perform_rolling_forecast
from scripts.forecast.interval_forecast_fixed import perform_interval_forecast

def compute_net_load(load_df, pv_df, wind_df, time_col='datetime', pred_col='predicted', lower_col='lower_bound', upper_col='upper_bound', actual_col='actual'):
    """计算净负荷及其区间和实际值"""
    df = load_df[[time_col, pred_col, lower_col, upper_col, actual_col]].rename(
        columns={pred_col: 'load_pred', lower_col: 'load_lower', upper_col: 'load_upper', actual_col: 'load_actual'}
    ).copy()
    df = df.merge(
        pv_df[[time_col, pred_col, lower_col, upper_col, actual_col]].rename(
            columns={pred_col: 'pv_pred', lower_col: 'pv_lower', upper_col: 'pv_upper', actual_col: 'pv_actual'}
        ),
        on=time_col, how='left'
    )
    df = df.merge(
        wind_df[[time_col, pred_col, lower_col, upper_col, actual_col]].rename(
            columns={pred_col: 'wind_pred', lower_col: 'wind_lower', upper_col: 'wind_upper', actual_col: 'wind_actual'}
        ),
        on=time_col, how='left'
    )
    df['net_load_pred'] = df['load_pred'] - df['pv_pred'].fillna(0) - df['wind_pred'].fillna(0)
    df['net_load_pred'] = df['net_load_pred'].clip(lower=0)
    df['net_load_lower'] = (df['load_lower'] - df['pv_upper'].fillna(0) - df['wind_upper'].fillna(0)).clip(lower=0)
    df['net_load_upper'] = (df['load_upper'] - df['pv_lower'].fillna(0) - df['wind_lower'].fillna(0)).clip(lower=0)
    if 'load_actual' in df and 'pv_actual' in df and 'wind_actual' in df:
        df['net_load_actual'] = df['load_actual'] - df['pv_actual'].fillna(0) - df['wind_actual'].fillna(0)
        df['net_load_actual'] = df['net_load_actual'].clip(lower=0)
    else:
        df['net_load_actual'] = None
    return df[[time_col, 'net_load_pred', 'net_load_lower', 'net_load_upper', 'net_load_actual']]

def calc_interval_metrics(net_load_df):
    df = net_load_df.dropna(subset=['net_load_pred', 'net_load_actual'])
    if len(df) == 0:
        return {}
    hit_mask = (df['net_load_actual'] >= df['net_load_lower']) & (df['net_load_actual'] <= df['net_load_upper'])
    hit_rate = hit_mask.mean() * 100
    mae = np.mean(np.abs(df['net_load_pred'] - df['net_load_actual']))
    mape = np.mean(np.abs((df['net_load_pred'] - df['net_load_actual']) / (df['net_load_actual'] + 1e-6))) * 100
    rmse = np.sqrt(np.mean((df['net_load_pred'] - df['net_load_actual']) ** 2))
    return {
        "hit_rate": hit_rate,
        "mae": mae,
        "mape": mape,
        "rmse": rmse
    }

def run_net_load_forecast(params: Dict[str, Any], mode: str = 'interval') -> Dict[str, Any]:
    """
    params: 预测参数字典，需包含province、日期、预测类型等
    mode: 'interval'/'day_ahead'/'rolling'
    返回: 与区间预测一致的json结构
    """
    province = params.get('province', '上海')
    forecast_type = params.get('forecastType', 'load')
    # 日期参数
    if mode == 'interval':
        # 区间预测
        start_date = params.get('forecastDate') or params.get('start_date')
        end_date = params.get('forecastEndDate') or params.get('end_date') or start_date
        confidence_level = float(params.get('confidenceLevel', 0.9))
        historical_days = int(params.get('historicalDays', 14))
        interval_minutes = int(params.get('interval', 15))
        # 1. 负荷
        load_df, _ = perform_interval_forecast(
            data_path=f"data/timeseries_load_{province}.csv",
            forecast_type='load',
            province=province,
            start_date_str=start_date,
            end_date_str=end_date,
            n_intervals=historical_days,
            model_path=f"models/convtrans_peak/load/{province}",
            scaler_path=f"models/scalers/convtrans_peak/load/{province}",
            device='cpu',
            quantiles=None,
            rolling=False,
            seq_length=96,
            peak_hours=(9, 20),
            valley_hours=(0, 6),
            fix_nan=True
        )
        # 2. 光伏
        pv_df, _ = perform_interval_forecast(
            data_path=f"data/timeseries_pv_{province}.csv",
            forecast_type='pv',
            province=province,
            start_date_str=start_date,
            end_date_str=end_date,
            n_intervals=historical_days,
            model_path=f"models/convtrans_peak/pv/{province}",
            scaler_path=f"models/scalers/convtrans_peak/pv/{province}",
            device='cpu',
            quantiles=None,
            rolling=False,
            seq_length=96,
            peak_hours=(9, 20),
            valley_hours=(0, 6),
            fix_nan=True
        )
        # 3. 风电
        wind_df, _ = perform_interval_forecast(
            data_path=f"data/timeseries_wind_{province}.csv",
            forecast_type='wind',
            province=province,
            start_date_str=start_date,
            end_date_str=end_date,
            n_intervals=historical_days,
            model_path=f"models/convtrans_peak/wind/{province}",
            scaler_path=f"models/scalers/convtrans_peak/wind/{province}",
            device='cpu',
            quantiles=None,
            rolling=False,
            seq_length=96,
            peak_hours=(9, 20),
            valley_hours=(0, 6),
            fix_nan=True
        )
    elif mode == 'day_ahead':
        # 日前预测
        forecast_date = params.get('forecastDate')
        forecast_end_date = params.get('forecastEndDate', forecast_date)
        historical_days = int(params.get('historicalDays', 8))
        # 1. 负荷
        load_df = perform_day_ahead_forecast(
            data_path=f"data/timeseries_load_{province}.csv",
            forecast_date=forecast_date,
            dataset_id=province,
            forecast_type='load',
            historical_days=historical_days,
            forecast_end_date=forecast_end_date
        )
        # 2. 光伏
        pv_df = perform_day_ahead_forecast(
            data_path=f"data/timeseries_pv_{province}.csv",
            forecast_date=forecast_date,
            dataset_id=province,
            forecast_type='pv',
            historical_days=historical_days,
            forecast_end_date=forecast_end_date
        )
        # 3. 风电
        wind_df = perform_day_ahead_forecast(
            data_path=f"data/timeseries_wind_{province}.csv",
            forecast_date=forecast_date,
            dataset_id=province,
            forecast_type='wind',
            historical_days=historical_days,
            forecast_end_date=forecast_end_date
        )
    elif mode == 'rolling':
        # 滚动预测
        start_date = params.get('startDate')
        end_date = params.get('endDate')
        interval = int(params.get('interval', 15))
        historical_days = int(params.get('historicalDays', 8))
        # 1. 负荷
        load_df = perform_rolling_forecast(
            data_path=f"data/timeseries_load_{province}.csv",
            start_date=start_date,
            end_date=end_date,
            forecast_interval=interval,
            dataset_id=province,
            forecast_type='load',
            historical_days=historical_days
        )
        # 2. 光伏
        pv_df = perform_rolling_forecast(
            data_path=f"data/timeseries_pv_{province}.csv",
            start_date=start_date,
            end_date=end_date,
            forecast_interval=interval,
            dataset_id=province,
            forecast_type='pv',
            historical_days=historical_days
        )
        # 3. 风电
        wind_df = perform_rolling_forecast(
            data_path=f"data/timeseries_wind_{province}.csv",
            start_date=start_date,
            end_date=end_date,
            forecast_interval=interval,
            dataset_id=province,
            forecast_type='wind',
            historical_days=historical_days
        )
    else:
        raise ValueError(f"未知的净负荷预测模式: {mode}")

    # 合成净负荷
    net_load_df = compute_net_load(load_df, pv_df, wind_df)
    metrics = calc_interval_metrics(net_load_df)

    # 生成json结构
    result = {
        "status": "success",
        "forecast_type": "net_load",
        "predictions": [
            {
                "datetime": row['datetime'],
                "predicted": row['net_load_pred'],
                "lower_bound": row['net_load_lower'],
                "upper_bound": row['net_load_upper'],
                "actual": row['net_load_actual']
            }
            for _, row in net_load_df.iterrows()
        ],
        "metrics": metrics
    }
    return result

# 命令行入口保留，支持csv输入
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="净负荷区间预测结果生成（支持函数式和命令行）")
    parser.add_argument('--mode', choices=['interval', 'day_ahead', 'rolling'], default='interval', help='预测模式')
    parser.add_argument('--province', required=True, help='省份')
    parser.add_argument('--forecastDate', help='预测日期/开始日期')
    parser.add_argument('--forecastEndDate', help='预测结束日期')
    parser.add_argument('--startDate', help='滚动预测开始日期')
    parser.add_argument('--endDate', help='滚动预测结束日期')
    parser.add_argument('--output_json', required=True, help='输出json文件路径')
    # 其他参数可按需添加
    args = parser.parse_args()
    # 构造参数字典
    params = vars(args)
    result = run_net_load_forecast(params, mode=args.mode)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"净负荷区间预测结果已保存到: {args.output_json}") 