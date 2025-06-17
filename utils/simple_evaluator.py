#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版评估器脚本，提供基本的评估指标计算和简单的可视化功能
不依赖复杂的外部库，便于在各种环境中使用
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(actual, predicted):
    """
    计算评估指标
    
    参数:
    actual: 实际值列表或数组
    predicted: 预测值列表或数组
    
    返回:
    包含各种评估指标的字典
    """
    # 检查输入数据
    if len(actual) == 0 or len(predicted) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'r2': np.nan,
            'smape': np.nan
        }
    
    # 将输入转换为numpy数组
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 过滤掉缺失值
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'r2': np.nan,
            'smape': np.nan
        }
    
    # 计算基本指标
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # 计算MAPE (避免除以零)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((actual - predicted) / actual) * 100
        mape = np.mean(mape_values[~np.isinf(mape_values) & ~np.isnan(mape_values)])
    
    # 计算R^2
    if len(actual) > 1:
        r2 = r2_score(actual, predicted)
    else:
        r2 = np.nan
    
    # 计算SMAPE (对称平均绝对百分比误差)
    with np.errstate(divide='ignore', invalid='ignore'):
        smape_values = 200 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
        smape = np.mean(smape_values[~np.isinf(smape_values) & ~np.isnan(smape_values)])
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'smape': smape
    }

def calculate_peak_metrics(df, peak_column='is_peak'):
    """
    分别计算高峰和非高峰时段的评估指标
    
    参数:
    df: 包含actual, predicted和peak_column列的DataFrame
    peak_column: 标识高峰时段的列名
    
    返回:
    包含高峰和非高峰时段评估指标的字典
    """
    # 确保必要的列存在
    required_cols = ['actual', 'predicted', peak_column]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame缺少必要的列: {missing}")
    
    # 过滤出有实际值的行
    valid_df = df.dropna(subset=['actual', 'predicted'])
    
    if len(valid_df) == 0:
        return {
            'overall': calculate_metrics([], []),
            'peak': calculate_metrics([], []),
            'non_peak': calculate_metrics([], [])
        }
    
    # 计算整体指标
    overall_metrics = calculate_metrics(valid_df['actual'], valid_df['predicted'])
    
    # 计算高峰时段指标
    peak_df = valid_df[valid_df[peak_column] == 1]
    peak_metrics = calculate_metrics(peak_df['actual'], peak_df['predicted'])
    
    # 计算非高峰时段指标
    non_peak_df = valid_df[valid_df[peak_column] == 0]
    non_peak_metrics = calculate_metrics(non_peak_df['actual'], non_peak_df['predicted'])
    
    return {
        'overall': overall_metrics,
        'peak': peak_metrics,
        'non_peak': non_peak_metrics
    }

def simple_plot_forecast(actual, predicted, dates=None, title="预测结果对比", save_path=None):
    """
    绘制实际值与预测值的对比图
    
    参数:
    actual: 实际值列表或数组
    predicted: 预测值列表或数组
    dates: 日期时间索引（可选）
    title: 图表标题
    save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    if dates is not None:
        plt.plot(dates, actual, 'b-', label='实际值')
        plt.plot(dates, predicted, 'r--', label='预测值')
    else:
        plt.plot(actual, 'b-', label='实际值')
        plt.plot(predicted, 'r--', label='预测值')
    
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('负荷 (MW)')
    plt.legend()
    plt.grid(True)
    
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    # else:
    #     plt.show()

def print_metrics_report(metrics, section_name="预测指标"):
    """
    打印评估指标报告
    
    参数:
    metrics: 包含评估指标的字典
    section_name: 报告的部分名称
    """
    print(f"\n===== {section_name} =====")
    print(f"MAE:   {metrics['mae']:.4f}")
    print(f"RMSE:  {metrics['rmse']:.4f}")
    print(f"MAPE:  {metrics['mape']:.2f}%")
    print(f"R²:    {metrics['r2']:.4f}")
    print(f"SMAPE: {metrics['smape']:.2f}%")

def simple_daily_peak_comparison(df, day_column='day', actual_col='actual', 
                                predicted_col='predicted', peak_col='is_peak',
                                title="每日高峰/非高峰时段预测对比", save_path=None):
    """
    对每天的高峰和非高峰时段预测结果进行对比分析
    
    参数:
    df: 包含预测结果的DataFrame
    day_column: 日期列名
    actual_col: 实际值列名
    predicted_col: 预测值列名
    peak_col: 高峰标识列名
    title: 图表标题
    save_path: 保存路径（可选）
    """
    # 确保必要的列存在
    required_cols = [day_column, actual_col, predicted_col, peak_col]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame缺少必要的列: {missing}")
    
    # 按天分组
    days = df[day_column].unique()
    
    if len(days) == 0:
        print("无法进行每日对比，数据中没有日期信息")
        return
    
    peak_mape_by_day = []
    non_peak_mape_by_day = []
    day_labels = []
    
    # 对每一天计算高峰和非高峰时段的MAPE
    for day in sorted(days):
        day_data = df[df[day_column] == day]
        
        # 高峰时段
        peak_data = day_data[day_data[peak_col] == 1]
        if len(peak_data) > 0:
            peak_metrics = calculate_metrics(peak_data[actual_col], peak_data[predicted_col])
            peak_mape_by_day.append(peak_metrics['mape'])
        else:
            peak_mape_by_day.append(np.nan)
        
        # 非高峰时段
        non_peak_data = day_data[day_data[peak_col] == 0]
        if len(non_peak_data) > 0:
            non_peak_metrics = calculate_metrics(non_peak_data[actual_col], non_peak_data[predicted_col])
            non_peak_mape_by_day.append(non_peak_metrics['mape'])
        else:
            non_peak_mape_by_day.append(np.nan)
        
        # 添加日期标签
        day_labels.append(str(day))
    
    # 绘制每日MAPE对比图
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.35
    index = np.arange(len(day_labels))
    
    # 绘制高峰和非高峰时段的MAPE
    plt.bar(index, peak_mape_by_day, bar_width, label='高峰时段', color='r', alpha=0.7)
    plt.bar(index + bar_width, non_peak_mape_by_day, bar_width, label='非高峰时段', color='b', alpha=0.7)
    
    plt.xlabel('日期')
    plt.ylabel('MAPE (%)')
    plt.title(title)
    plt.xticks(index + bar_width/2, day_labels, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    # else:
    #     plt.show()
    
    # 返回每日指标摘要
    return {
        'day_labels': day_labels,
        'peak_mape': peak_mape_by_day,
        'non_peak_mape': non_peak_mape_by_day
    }

def simple_peak_analysis(df, actual_col='actual', predicted_col='predicted', peak_col='is_peak',
                        title="高峰时段分析", save_path=None):
    """
    分析高峰和非高峰时段的预测性能
    
    参数:
    df: 包含预测结果的DataFrame
    actual_col: 实际值列名
    predicted_col: 预测值列名
    peak_col: 高峰标识列名
    title: 图表标题
    save_path: 保存路径（可选）
    """
    # 确保必要的列存在
    required_cols = [actual_col, predicted_col, peak_col]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame缺少必要的列: {missing}")
    
    # 过滤出有实际值的行
    valid_df = df.dropna(subset=[actual_col, predicted_col])
    
    if len(valid_df) == 0:
        print("没有有效数据进行分析")
        return
    
    # 计算误差
    valid_df['error'] = valid_df[actual_col] - valid_df[predicted_col]
    valid_df['error_pct'] = valid_df['error'] / valid_df[actual_col] * 100
    
    # 分离高峰和非高峰时段
    peak_df = valid_df[valid_df[peak_col] == 1]
    non_peak_df = valid_df[valid_df[peak_col] == 0]
    
    # 计算指标
    peak_metrics = calculate_metrics(peak_df[actual_col], peak_df[predicted_col]) if len(peak_df) > 0 else None
    non_peak_metrics = calculate_metrics(non_peak_df[actual_col], non_peak_df[predicted_col]) if len(non_peak_df) > 0 else None
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 散点图: 高峰时段
    if len(peak_df) > 0:
        axes[0, 0].scatter(peak_df[actual_col], peak_df[predicted_col], color='red', alpha=0.7)
        min_val = min(peak_df[actual_col].min(), peak_df[predicted_col].min())
        max_val = max(peak_df[actual_col].max(), peak_df[predicted_col].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--')
        axes[0, 0].set_title('高峰时段')
        axes[0, 0].set_xlabel('实际值')
        axes[0, 0].set_ylabel('预测值')
        
        # 添加指标文本
        if peak_metrics:
            metrics_text = f'MAE: {peak_metrics["mae"]:.2f}\nRMSE: {peak_metrics["rmse"]:.2f}\nMAPE: {peak_metrics["mape"]:.2f}%'
            axes[0, 0].text(0.05, 0.95, metrics_text, transform=axes[0, 0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[0, 0].text(0.5, 0.5, '无高峰时段数据', ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # 2. 散点图: 非高峰时段
    if len(non_peak_df) > 0:
        axes[0, 1].scatter(non_peak_df[actual_col], non_peak_df[predicted_col], color='blue', alpha=0.7)
        min_val = min(non_peak_df[actual_col].min(), non_peak_df[predicted_col].min())
        max_val = max(non_peak_df[actual_col].max(), non_peak_df[predicted_col].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--')
        axes[0, 1].set_title('非高峰时段')
        axes[0, 1].set_xlabel('实际值')
        axes[0, 1].set_ylabel('预测值')
        
        # 添加指标文本
        if non_peak_metrics:
            metrics_text = f'MAE: {non_peak_metrics["mae"]:.2f}\nRMSE: {non_peak_metrics["rmse"]:.2f}\nMAPE: {non_peak_metrics["mape"]:.2f}%'
            axes[0, 1].text(0.05, 0.95, metrics_text, transform=axes[0, 1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[0, 1].text(0.5, 0.5, '无非高峰时段数据', ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. 误差分布: 高峰时段
    if len(peak_df) > 0:
        axes[1, 0].hist(peak_df['error_pct'], bins=20, color='red', alpha=0.7)
        axes[1, 0].axvline(x=0, color='k', linestyle='--')
        axes[1, 0].set_title('高峰时段误差分布')
        axes[1, 0].set_xlabel('误差百分比 (%)')
        axes[1, 0].set_ylabel('频率')
    else:
        axes[1, 0].text(0.5, 0.5, '无高峰时段数据', ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. 误差分布: 非高峰时段
    if len(non_peak_df) > 0:
        axes[1, 1].hist(non_peak_df['error_pct'], bins=20, color='blue', alpha=0.7)
        axes[1, 1].axvline(x=0, color='k', linestyle='--')
        axes[1, 1].set_title('非高峰时段误差分布')
        axes[1, 1].set_xlabel('误差百分比 (%)')
        axes[1, 1].set_ylabel('频率')
    else:
        axes[1, 1].text(0.5, 0.5, '无非高峰时段数据', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    # else:
    #     plt.show()
    
    # 返回指标对比
    return {
        'peak': peak_metrics,
        'non_peak': non_peak_metrics
    }

def simple_error_histogram(actual, predicted, bins=20, title="误差分布", save_path=None):
    """
    绘制误差分布直方图
    
    参数:
    actual: 实际值列表或数组
    predicted: 预测值列表或数组
    bins: 柱状图的柱子数量
    title: 图表标题
    save_path: 保存路径（可选）
    """
    # 计算绝对误差和百分比误差
    errors = np.array(actual) - np.array(predicted)
    
    # 计算百分比误差 (避免除以零)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_errors = errors / np.array(actual) * 100
        pct_errors = pct_errors[~np.isinf(pct_errors) & ~np.isnan(pct_errors)]
    
    plt.figure(figsize=(12, 5))
    
    # 创建两个子图
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=bins, color='blue', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('绝对误差分布')
    plt.xlabel('误差')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(pct_errors, bins=bins, color='green', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('百分比误差分布')
    plt.xlabel('误差百分比 (%)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    # else:
    #     plt.show()

# 测试代码
if __name__ == "__main__":
    # 示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=96, freq='15min')
    actual = 10000 + 2000 * np.sin(np.linspace(0, 2*np.pi, 96)) + np.random.normal(0, 200, 96)
    predicted = actual + np.random.normal(0, 500, 96)
    
    # 设置高峰时段
    is_peak = np.zeros(96, dtype=int)
    is_peak[32:64] = 1  # 假设8点到16点是高峰时段
    
    # 创建测试DataFrame
    test_df = pd.DataFrame({
        'datetime': dates,
        'actual': actual,
        'predicted': predicted,
        'is_peak': is_peak
    })
    
    # 测试函数
    metrics = simple_peak_analysis(
        test_df,
        title='简化评估器测试',
        save_path='results/simple_evaluator_test.png'
    ) 