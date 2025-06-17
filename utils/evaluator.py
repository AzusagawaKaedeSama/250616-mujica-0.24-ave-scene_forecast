import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils.plot_style

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

def plot_peak_forecast_analysis(results_df, title="高峰时段预测分析", save_path=None):
    """分析高峰时段和非高峰时段的预测性能差异"""
    
    # 添加数据有效性检查
    # 检查是否有足够的有效数据进行绘图
    has_valid_data = (
        not results_df.empty and 
        'actual' in results_df.columns and 
        'predicted' in results_df.columns and
        'is_peak' in results_df.columns and
        results_df['actual'].notna().sum() > 0 and
        results_df['predicted'].notna().sum() > 0
    )
    
    if not has_valid_data:
        print("警告: 没有足够的有效数据进行高峰预测分析")
        if save_path:
            # 创建一个简单的图表说明没有有效数据
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "没有足够的有效数据进行分析\n可能原因：没有实际观测值或预测值全为NaN", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"已保存空白分析图表到 {save_path}")
        return
    
    # 添加误差百分比列
    valid_mask = (results_df['actual'].notna()) & (results_df['predicted'].notna()) & (results_df['actual'] != 0)
    
    # 如果没有有效的误差计算点，提前返回
    if valid_mask.sum() == 0:
        print("警告: 无法计算误差百分比，可能是缺少有效的实际值")
        if save_path:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "无法计算误差百分比\n可能原因：缺少实际观测值或存在除以零的情况", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"已保存空白分析图表到 {save_path}")
        return
    
    results_df.loc[valid_mask, 'error_pct'] = (
        (results_df.loc[valid_mask, 'predicted'] - results_df.loc[valid_mask, 'actual']) / 
        results_df.loc[valid_mask, 'actual'] * 100
    )
    
    # 创建高峰和非高峰子集
    peak_mask = valid_mask & (results_df['is_peak'] == 1)
    non_peak_mask = valid_mask & (results_df['is_peak'] == 0)
    
    # 如果任一子集为空，打印警告并提前返回
    if peak_mask.sum() == 0 or non_peak_mask.sum() == 0:
        print("警告: 高峰或非高峰时段缺少有效数据")
        if save_path:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "高峰或非高峰时段缺少有效数据进行比较分析", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"已保存空白分析图表到 {save_path}")
        return
    
    peak_df = results_df[peak_mask].copy()
    non_peak_df = results_df[non_peak_mask].copy()
    
    # 计算指标
    peak_metrics = calculate_metrics(peak_df['actual'], peak_df['predicted'])
    non_peak_metrics = calculate_metrics(non_peak_df['actual'], non_peak_df['predicted'])
    all_metrics = calculate_metrics(results_df.loc[valid_mask, 'actual'], results_df.loc[valid_mask, 'predicted'])
    
    # 创建图表
    plt.figure(figsize=(16, 12))
    plt.suptitle(title, fontsize=16)
    
    gs = gridspec.GridSpec(2, 3)
    
    # 图1: 预测vs实际 (全部)
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(results_df.loc[peak_mask, 'actual'], results_df.loc[peak_mask, 'predicted'], 
               alpha=0.6, label='高峰时段', color='red')
    ax1.scatter(results_df.loc[non_peak_mask, 'actual'], results_df.loc[non_peak_mask, 'predicted'], 
                alpha=0.6, label='非高峰时段', color='blue')
    
    # 添加对角线
    min_val = min(results_df.loc[valid_mask, 'actual'].min(), results_df.loc[valid_mask, 'predicted'].min())
    max_val = max(results_df.loc[valid_mask, 'actual'].max(), results_df.loc[valid_mask, 'predicted'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    ax1.set_xlabel('实际值')
    ax1.set_ylabel('预测值')
    ax1.set_title('预测vs实际')
    ax1.legend()
    ax1.grid(True)
    
    # 图2: 误差百分比随时间变化
    ax2 = plt.subplot(gs[0, 1:])
    times = pd.to_datetime(results_df.loc[valid_mask, 'datetime'])
    peak_times = pd.to_datetime(peak_df['datetime'])
    non_peak_times = pd.to_datetime(non_peak_df['datetime'])
    
    # 确保存在有效的时间点和误差数据
    if not times.empty and 'error_pct' in results_df.columns:
        ax2.plot(times, results_df.loc[valid_mask, 'error_pct'], 'k-', alpha=0.3, label='全部')
        ax2.scatter(peak_times, peak_df['error_pct'], color='red', alpha=0.6, label='高峰时段')
        ax2.scatter(non_peak_times, non_peak_df['error_pct'], color='blue', alpha=0.6, label='非高峰时段')
        ax2.axhline(0, color='green', linestyle='--')
        
        # 添加小时标记
        ax2.set_xlabel('时间')
        ax2.set_ylabel('误差百分比 (%)')
        ax2.set_title('误差百分比随时间变化')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, "没有有效的时间序列误差数据", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12)
    
    # 图3: 绝对误差箱形图比较
    ax3 = plt.subplot(gs[1, 0])
    
    # 确保有足够数据绘制箱形图
    if len(peak_df) > 1 and len(non_peak_df) > 1:
        box_data = [
            peak_df['error_pct'].abs(),
            non_peak_df['error_pct'].abs()
        ]
        ax3.boxplot(box_data, labels=['高峰时段', '非高峰时段'])
        ax3.set_ylabel('绝对误差百分比 (%)')
        ax3.set_title('绝对误差分布比较')
        ax3.grid(True, axis='y')
    else:
        ax3.text(0.5, 0.5, "没有足够数据绘制箱形图", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes, fontsize=12)
    
    # 图4: 误差直方图对比
    ax4 = plt.subplot(gs[1, 1])
    
    # 确保有数据且不全是NaN
    if not peak_df['error_pct'].isna().all() and not non_peak_df['error_pct'].isna().all():
        try:
            ax4.hist(peak_df['error_pct'], bins=20, alpha=0.5, color='red', label='高峰时段')
            ax4.hist(non_peak_df['error_pct'], bins=20, alpha=0.5, color='blue', label='非高峰时段')
            ax4.set_xlabel('误差百分比 (%)')
            ax4.set_ylabel('频次')
            ax4.set_title('误差分布直方图')
            ax4.legend()
            ax4.grid(True)
        except Exception as e:
            print(f"绘制直方图时出错: {e}")
            ax4.text(0.5, 0.5, f"无法绘制直方图: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=10)
    else:
        ax4.text(0.5, 0.5, "没有有效数据绘制直方图", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=12)
    
    # 图5: 指标对比表格
    ax5 = plt.subplot(gs[1, 2])
    ax5.axis('off')
    
    metrics_table = [
        ['指标', '全部', '高峰时段', '非高峰时段'],
        ['MAE', f"{all_metrics['mae']:.4f}", f"{peak_metrics['mae']:.4f}", f"{non_peak_metrics['mae']:.4f}"],
        ['RMSE', f"{all_metrics['rmse']:.4f}", f"{peak_metrics['rmse']:.4f}", f"{non_peak_metrics['rmse']:.4f}"],
        ['MAPE', f"{all_metrics['mape']:.4f}%", f"{peak_metrics['mape']:.4f}%", f"{non_peak_metrics['mape']:.4f}%"],
        ['R²', f"{all_metrics['r2']:.4f}", f"{peak_metrics['r2']:.4f}", f"{non_peak_metrics['r2']:.4f}"]
    ]
    
    table = ax5.table(cellText=metrics_table, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # 为表格第一行添加颜色
    for j, cell in table._cells.items():
        if j[0] == 0:  # 第一行
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
    
    ax5.set_title('性能指标对比')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭而不是显示

def calculate_metrics(actual, predicted):
    """计算各种评估指标"""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # 计算R^2
    if np.var(actual) == 0:
        r2 = 0
    else:
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    
    # 计算SMAPE (对称平均绝对百分比误差)
    smape = 200 * np.mean(np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'smape': smape
    }

def plot_forecast_results(results_df, results_dir, start_datetime, end_datetime, timestamp, peak_hours):
    """绘制预测结果可视化图表"""
    plt.figure(figsize=(14, 10))
    
    # 创建上下两个子图
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    # 上面的子图：预测结果对比
    ax1 = plt.subplot(gs[0])
    
    # 绘制实际值
    ax1.plot(results_df['datetime'], results_df['actual'], 'b-', label='实际值', linewidth=2)
    
    # 绘制预测值
    if 'predicted_smoothed' in results_df.columns and not results_df['predicted_smoothed'].equals(results_df['predicted']):
        ax1.plot(results_df['datetime'], results_df['predicted'], 'r--', label='原始预测值', linewidth=1, alpha=0.6)
        ax1.plot(results_df['datetime'], results_df['predicted_smoothed'], 'g-', label='平滑预测值', linewidth=1.5)
    else:
        ax1.plot(results_df['datetime'], results_df['predicted'], 'r--', label='预测值', linewidth=1.5)
    
    # 标记高峰时段
    peak_df = results_df[results_df['is_peak']]
    if len(peak_df) > 0:
        # 查找连续的高峰时段
        peak_periods = []
        start_idx = None
        
        for i, (idx, row) in enumerate(results_df.iterrows()):
            if row['is_peak'] and start_idx is None:
                # 找到高峰开始
                start_idx = i
            elif not row['is_peak'] and start_idx is not None:
                # 找到高峰结束
                peak_periods.append((start_idx, i-1))
                start_idx = None
        
        # 检查最后一个高峰区间
        if start_idx is not None:
            peak_periods.append((start_idx, len(results_df)-1))
        
        # 为每个高峰时段添加背景
        for start_idx, end_idx in peak_periods:
            ax1.axvspan(
                results_df['datetime'].iloc[start_idx],
                results_df['datetime'].iloc[end_idx],
                color='yellow', alpha=0.2, label='_' if len(peak_periods) > 1 else '高峰时段'
            )
    
    ax1.set_title('高峰感知负荷预测结果对比')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('负荷 (MW)')
    ax1.grid(True)
    ax1.legend()
    
    # 下面的子图：预测误差
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # 计算预测误差
    valid_df = results_df.dropna(subset=['actual'])
    if len(valid_df) > 0:
        valid_df['error'] = valid_df['actual'] - valid_df['predicted']
        
        if 'predicted_smoothed' in valid_df.columns and not valid_df['predicted_smoothed'].equals(valid_df['predicted']):
            valid_df['error_smoothed'] = valid_df['actual'] - valid_df['predicted_smoothed']
        
        # 分别绘制高峰和非高峰时段的误差
        peak_valid_df = valid_df[valid_df['is_peak']]
        non_peak_valid_df = valid_df[~valid_df['is_peak']]
        
        if len(peak_valid_df) > 0:
            ax2.scatter(peak_valid_df['datetime'], peak_valid_df['error'], 
                       color='red', marker='o', s=20, alpha=0.7, label='高峰时段误差')
            
            if 'error_smoothed' in peak_valid_df.columns:
                ax2.scatter(peak_valid_df['datetime'], peak_valid_df['error_smoothed'], 
                           color='darkred', marker='x', s=15, alpha=0.7, label='高峰时段平滑误差')
        
        if len(non_peak_valid_df) > 0:
            ax2.scatter(non_peak_valid_df['datetime'], non_peak_valid_df['error'], 
                       color='blue', marker='o', s=20, alpha=0.5, label='非高峰时段误差')
            
            if 'error_smoothed' in non_peak_valid_df.columns:
                ax2.scatter(non_peak_valid_df['datetime'], non_peak_valid_df['error_smoothed'], 
                           color='darkblue', marker='x', s=15, alpha=0.5, label='非高峰时段平滑误差')
        
        # 添加零线
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # 为每个工作日添加竖线
        for date in pd.date_range(start=start_datetime.date(), end=end_datetime.date()):
            if date.dayofweek < 5:  # 周一至周五
                ax2.axvline(x=pd.Timestamp(date), color='gray', linestyle='--', alpha=0.3)
    
    ax2.set_title('预测误差分析')
    ax2.set_xlabel('时间')
    ax2.set_ylabel('误差 (MW)')
    ax2.grid(True)
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plot_path = f"{results_dir}/peak_aware_forecast_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到 {plot_path}")
    
    # 创建每日高峰vs非高峰对比图
    plot_daily_peak_comparison(results_df, results_dir, start_datetime, end_datetime, timestamp, peak_hours)

def plot_daily_peak_comparison(results_df, results_dir, start_datetime, end_datetime, timestamp, peak_hours):
    """为每天绘制高峰与非高峰时段的对比图"""
    # 获取预测的日期范围
    dates = pd.date_range(start=start_datetime.date(), end=end_datetime.date())
    
    # 仅分析有实际值的数据
    valid_df = results_df.dropna(subset=['actual']).copy()
    if len(valid_df) == 0:
        return
    
    # 添加日期列
    valid_df['date'] = valid_df['datetime'].dt.date
    
    # 创建指标容器
    daily_metrics = []
    
    # 计算每天的高峰和非高峰指标
    for date in dates:
        day_data = valid_df[valid_df['date'] == date.date()]
        
        if len(day_data) > 0:
            # 分离高峰和非高峰数据
            day_peak = day_data[day_data['is_peak']]
            day_non_peak = day_data[~day_data['is_peak']]
            
            # 计算当天指标
            day_metrics = {
                'date': date.date(),
                'day_of_week': date.dayofweek,
                'is_weekend': date.dayofweek >= 5,
                'total_points': len(day_data),
                'peak_points': len(day_peak),
                'non_peak_points': len(day_non_peak)
            }
            
            # 高峰时段指标
            if len(day_peak) > 0:
                peak_metrics = calculate_metrics(day_peak['actual'], day_peak['predicted'])
                for key, value in peak_metrics.items():
                    day_metrics[f'peak_{key}'] = value
            
            # 非高峰时段指标
            if len(day_non_peak) > 0:
                non_peak_metrics = calculate_metrics(day_non_peak['actual'], day_non_peak['predicted'])
                for key, value in non_peak_metrics.items():
                    day_metrics[f'non_peak_{key}'] = value
            
            # 全天指标
            overall_metrics = calculate_metrics(day_data['actual'], day_data['predicted'])
            for key, value in overall_metrics.items():
                day_metrics[f'overall_{key}'] = value
            
            daily_metrics.append(day_metrics)
    
    # 如果没有每日指标，返回
    if not daily_metrics:
        return
    
    # 创建每日指标DataFrame
    daily_df = pd.DataFrame(daily_metrics)
    
    # 保存每日指标
    daily_metrics_path = f"{results_dir}/daily_metrics_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.csv"
    daily_df.to_csv(daily_metrics_path, index=False)
    
    # 绘制每日指标对比图
    if len(daily_df) > 0:
        plt.figure(figsize=(14, 10))
        
        # 1. MAPE对比
        plt.subplot(2, 2, 1)
        plt.bar(range(len(daily_df)), daily_df['overall_mape'], color='gray', alpha=0.6, label='全天')
        
        if 'peak_mape' in daily_df.columns:
            plt.bar(range(len(daily_df)), daily_df['peak_mape'], color='red', alpha=0.6, label='高峰时段')
        
        if 'non_peak_mape' in daily_df.columns:
            plt.bar(range(len(daily_df)), daily_df['non_peak_mape'], color='blue', alpha=0.6, label='非高峰时段')
        
        plt.xticks(range(len(daily_df)), [d.strftime('%m-%d') for d in daily_df['date']], rotation=45)
        plt.title('每日MAPE对比')
        plt.ylabel('MAPE (%)')
        plt.legend()
        plt.grid(axis='y')
        
        # 2. MAE对比
        plt.subplot(2, 2, 2)
        plt.bar(range(len(daily_df)), daily_df['overall_mae'], color='gray', alpha=0.6, label='全天')
        
        if 'peak_mae' in daily_df.columns:
            plt.bar(range(len(daily_df)), daily_df['peak_mae'], color='red', alpha=0.6, label='高峰时段')
        
        if 'non_peak_mae' in daily_df.columns:
            plt.bar(range(len(daily_df)), daily_df['non_peak_mae'], color='blue', alpha=0.6, label='非高峰时段')
        
        plt.xticks(range(len(daily_df)), [d.strftime('%m-%d') for d in daily_df['date']], rotation=45)
        plt.title('每日MAE对比')
        plt.ylabel('MAE (MW)')
        plt.legend()
        plt.grid(axis='y')
        
        # 3. 数据点数量
        plt.subplot(2, 2, 3)
        plt.bar(range(len(daily_df)), daily_df['total_points'], color='gray', alpha=0.6, label='总数据点')
        plt.bar(range(len(daily_df)), daily_df['peak_points'], color='red', alpha=0.6, label='高峰时段点数')
        plt.bar(range(len(daily_df)), daily_df['non_peak_points'], color='blue', alpha=0.6, label='非高峰时段点数')
        
        plt.xticks(range(len(daily_df)), [d.strftime('%m-%d') for d in daily_df['date']], rotation=45)
        plt.title('每日数据点数量')
        plt.ylabel('数据点数')
        plt.legend()
        plt.grid(axis='y')
        
        # 4. R²对比
        plt.subplot(2, 2, 4)
        plt.bar(range(len(daily_df)), daily_df['overall_r2'], color='gray', alpha=0.6, label='全天')
        
        if 'peak_r2' in daily_df.columns:
            plt.bar(range(len(daily_df)), daily_df['peak_r2'], color='red', alpha=0.6, label='高峰时段')
        
        if 'non_peak_r2' in daily_df.columns:
            plt.bar(range(len(daily_df)), daily_df['non_peak_r2'], color='blue', alpha=0.6, label='非高峰时段')
        
        plt.xticks(range(len(daily_df)), [d.strftime('%m-%d') for d in daily_df['date']], rotation=45)
        plt.title('每日R²对比')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        daily_plot_path = f"{results_dir}/daily_metrics_plot_{start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}_{timestamp}.png"
        plt.savefig(daily_plot_path, dpi=300, bbox_inches='tight')
        print(f"每日指标图表已保存到 {daily_plot_path}")

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """计算基础评估指标"""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100 if np.sum(y_true) > 0 else 0
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    @staticmethod
    def plot_comparison(y_true, y_pred, title='预测结果对比'):
        """基础预测结果对比图"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='真实值', alpha=0.7, linewidth=2)
        plt.plot(y_pred, '--', label='预测值', linewidth=2)
        plt.title(title)
        plt.xlabel('时间步')
        plt.ylabel('负荷 (MW)')
        plt.legend()
        plt.grid(True)
        plt.close('all') # Close figure before show/save
        # if save_path:
        #     plt.savefig(save_path, bbox_inches='tight')
        # plt.show()

    @staticmethod
    def multi_model_comparison(models_data, true_values, 
                            model_names=None, 
                            title='多模型预测对比',
                            save_path=None):
        """多模型预测对比图（合并到同一张图）"""
        plt.figure(figsize=(14, 7))
        
        # 绘制真实值（仅画一次）
        plt.plot(true_values, label='真实值', 
                color='black', alpha=0.9, linewidth=3, zorder=100)
        
        # 定义颜色和线型列表
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
        linestyles = ['--', '-.', ':', '-']
        
        # 绘制每个模型的预测值
        for i, pred in enumerate(models_data):
            model_name = model_names[i] if model_names else f'Model {i+1}'
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            plt.plot(pred, 
                    linestyle=linestyle, 
                    linewidth=2.5,
                    color=color,
                    alpha=0.8,
                    label=model_name)
        
        plt.title(title, fontsize=14)
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('负荷 (MW)', fontsize=12)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.close('all') # Close figure before show/save
        # if save_path:
        #     plt.savefig(save_path, bbox_inches='tight')
        # plt.show()

    @staticmethod
    def accuracy_profile(y_true, predictions_list, 
                        threshold=10,  # 默认阈值设为10%
                        window_size=24, 
                        save_path=None):
        """预测准确率热力图（修正版）"""
        num_points = len(y_true) - window_size + 1
        num_models = len(predictions_list)
        
        # 初始化准确率矩阵 (时间窗口数 x 模型数)
        window_accuracy = np.zeros((num_points, num_models))
        
        # 计算每个窗口每个模型的准确率
        for i in range(num_points):
            for model_idx in range(num_models):
                # 提取当前窗口的预测和真实值
                window_true = y_true[i:i+window_size]
                window_pred = predictions_list[model_idx][i:i+window_size]
                
                # 计算误差百分比
                errors = np.abs((window_pred - window_true) / window_true) * 100
                # 统计准确率（误差 < threshold）
                accurate = np.sum(errors < threshold) / window_size
                window_accuracy[i, model_idx] = accurate
        
        # 计算平均准确率（按模型）
        avg_accuracy = np.mean(window_accuracy, axis=1)
        
        # 绘制热力图
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            avg_accuracy.reshape(-1, 1).T,  # 转换为 (1, num_points) 形状
            annot=False,
            cmap='YlGn',
            vmin=0, vmax=1,
            xticklabels=np.arange(num_points),
            yticklabels=['平均准确率']
        )
        plt.title(f'预测准确率热力图（阈值={threshold}%）')
        plt.xlabel('时间窗口起始点')
        plt.ylabel('')
        plt.close('all') # Close figure before show/save
        # if save_path:
        #     plt.savefig(save_path, bbox_inches='tight')
        # plt.show()

    @staticmethod
    def error_distribution(models_predictions, true_values, model_names, save_path=None):
        # 示例输入数据检查
        print(f"true_values 形状: {true_values.shape}")  # 应为 (n_samples,)
        print(f"第一个模型预测形状: {models_predictions[0].shape}")  # 应为 (n_samples,)# 展平所有数组并计算误差
        errors = []
        for pred, name in zip(models_predictions, model_names):
            # 确保 pred 和 true_values 是二维数组
            pred = np.array(pred).reshape(-1, 1)
            true = np.array(true_values).reshape(-1, 1)
            # 计算误差并展平为一维
            error = (true - pred).flatten()
            # 为每个误差添加模型标签
            errors.extend([(name, e) for e in error])
        
        # 转换为 DataFrame
        errors_df = pd.DataFrame(errors, columns=['Model', 'Error'])
        
        # 绘制箱型图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Error', data=errors_df)
        plt.title('模型误差分布对比')
        plt.close('all') # Close figure before show/save
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    @staticmethod
    def residual_analysis(y_true, y_pred, 
                        model_name='Model', 
                        save_path=None):
        """残差分析图"""
        # 确保输入为一维数组
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        residuals = y_pred - y_true
        
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=y_true, y=residuals, 
                        alpha=0.7,
                        label='残差分布',
                        s=80)
        plt.axhline(0, color='r', linestyle='--', linewidth=2)
        plt.title(f'{model_name} 残差分析')
        plt.xlabel('真实值')
        plt.ylabel('残差')
        plt.legend()
        plt.grid(True)
        plt.close('all') # Close figure before show/save
        # if save_path:
        #     plt.savefig(save_path, bbox_inches='tight')
        # plt.show()


def plot_regional_data(regional_data, title, save_path):
    """Plot regional load data."""
    plt.figure(figsize=(12, 8))
    
    for region, data in regional_data.items():
        plt.plot(data.index, data['load'], label=region)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_evaluation_radar(evaluation_results, save_path):
    """Plot radar chart of evaluation indices."""
    # Extract primary indices for each region
    regions = list(evaluation_results.keys())
    indices = ['ForecastReliability', 'ProvincialLoadImpact', 'ForecastingComplexity']
    
    # Create data for radar chart
    values = np.zeros((len(regions), len(indices)))
    for i, region in enumerate(regions):
        for j, index in enumerate(indices):
            values[i, j] = evaluation_results[region]['indices'][index]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(indices), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, region in enumerate(regions):
        values_closed = values[i].tolist()
        values_closed += values_closed[:1]  # Close the loop
        ax.plot(angles, values_closed, linewidth=2, label=region)
        ax.fill(angles, values_closed, alpha=0.1)
    
    # Set chart properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indices)
    ax.set_title('Evaluation Indices for Regions', size=15)
    ax.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_weights(weights, save_path):
    """Plot PCA-derived weights for regions."""
    plt.figure(figsize=(10, 6))
    
    regions = list(weights.keys())
    weight_values = [weights[region] for region in regions]
    
    bars = plt.bar(regions, weight_values, color='royalblue')
    
    # Add weight values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.title('PCA-Derived Weights for Regions')
    plt.ylabel('Weight')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_comparison(actual, direct, fusion, save_path):
    """Plot comparison of forecasting methods."""
    plt.figure(figsize=(14, 8))
    
    # Ensure we have data to plot
    if actual.empty or direct.empty or fusion.empty:
        print(f"Warning: Empty data for comparison plot. Skipping {save_path}")
        return
    
    # Slice to 1 day for better visualization if we have enough data
    if len(actual) > 96:  # Only slice if we have enough data
        start_idx = len(actual) // 2
        end_idx = min(start_idx + 96, len(actual))  # 1 day (assuming 15-min intervals)
    else:
        start_idx = 0
        end_idx = len(actual)
    
    plt.plot(actual.index[start_idx:end_idx], 
             actual['load'][start_idx:end_idx], 
             'k-', label='Actual', linewidth=2)
    
    plt.plot(direct.index[start_idx:end_idx], 
             direct['load'][start_idx:end_idx], 
             'r--', label='Direct Aggregation', linewidth=2)
    
    plt.plot(fusion.index[start_idx:end_idx], 
             fusion['load'][start_idx:end_idx], 
             'b-.', label='Weighted Fusion', linewidth=2)
    
    plt.title('Comparison of Forecast Integration Methods')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error_distribution(actual, direct, fusion, save_path):
    """Plot error distribution comparison."""
    # Calculate APEs
    direct_ape = np.abs((actual['load'] - direct['load']) / actual['load']) * 100
    fusion_ape = np.abs((actual['load'] - fusion['load']) / actual['load']) * 100
    
    # Create dataframe for plotting
    error_df = pd.DataFrame({
        'Direct Aggregation': direct_ape,
        'Weighted Fusion': fusion_ape
    })
    
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(data=error_df)
    plt.title('Error Distribution Comparison')
    plt.ylabel('Absolute Percentage Error (%)')
    plt.tight_layout()
    plt.close('all') # Close figure before show/save
    # if save_path:
    #     plt.savefig(save_path)
    # plt.show()


def plot_performance_metrics(direct_metrics, fusion_metrics, save_path):
    """Plot performance metrics comparison."""
    plt.figure(figsize=(12, 6))
    
    metrics = list(direct_metrics.keys())
    x = np.arange(len(metrics))
    width = 0.35
    
    direct_values = [direct_metrics[metric] for metric in metrics]
    fusion_values = [fusion_metrics[metric] for metric in metrics]
    
    bars1 = plt.bar(x - width/2, direct_values, width, label='Direct Aggregation')
    bars2 = plt.bar(x + width/2, fusion_values, width, label='Weighted Fusion')
    
    # Add values on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom', fontsize=8)
    
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def setup_directories():
    """创建必要的输出目录"""
    directories = ['results/multi_regional', 'results/fusion', 'models/gru', 'models/lstm']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)