#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成2024年各省典型天气场景日汇总表
"""

import os
import json
import pandas as pd
from datetime import datetime
import sys
# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import utils.plot_style

def load_analysis_results():
    """加载所有省份的分析结果"""
    results_dir = "results/weather_scenario_analysis/2024"
    provinces = ['上海', '江苏', '浙江', '安徽', '福建']
    
    all_results = {}
    
    for province in provinces:
        json_file = f"{results_dir}/{province}_analysis_results.json"
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                all_results[province] = json.load(f)
    
    return all_results

def generate_typical_days_summary(all_results):
    """生成典型场景日汇总表"""
    summary_data = []
    
    for province, data in all_results.items():
        typical_days = data.get('typical_days', {})
        
        for scenario_name, scenario_data in typical_days.items():
            characteristics = scenario_data['characteristics']
            typical_dates = scenario_data['typical_dates']
            
            for date in typical_dates:
                summary_data.append({
                    '省份': province,
                    '场景类型': scenario_name,
                    '典型日期': date,
                    '出现天数': scenario_data['count'],
                    '占比(%)': f"{scenario_data['percentage']:.1f}",
                    '平均温度(°C)': f"{characteristics['temperature_mean']:.1f}",
                    '最高温度(°C)': f"{characteristics['temperature_max']:.1f}",
                    '最低温度(°C)': f"{characteristics['temperature_min']:.1f}",
                    '平均湿度(%)': f"{characteristics['humidity_mean']:.1f}",
                    '平均风速(m/s)': f"{characteristics['wind_speed_mean']:.1f}",
                    '日降水量(mm)': f"{characteristics['precipitation_sum']:.1f}"
                })
    
    return pd.DataFrame(summary_data)

def generate_extreme_weather_summary(all_results):
    """生成极端天气事件汇总"""
    extreme_summary = []
    
    for province, data in all_results.items():
        extreme_days = data.get('extreme_days', [])
        
        # 统计各类极端天气的出现次数
        condition_counts = {}
        for day in extreme_days:
            for condition in day['conditions']:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        # 找出最严重的极端天气日（条件最多的）
        most_extreme_day = None
        max_conditions = 0
        for day in extreme_days:
            if len(day['conditions']) > max_conditions:
                max_conditions = len(day['conditions'])
                most_extreme_day = day
        
        extreme_summary.append({
            '省份': province,
            '极端天气总天数': len(extreme_days),
            '主要极端天气类型': ', '.join(sorted(condition_counts.keys(), key=condition_counts.get, reverse=True)[:3]),
            '最严重极端天气日': most_extreme_day['date'] if most_extreme_day else 'N/A',
            '最严重天气条件': ', '.join(most_extreme_day['conditions']) if most_extreme_day else 'N/A',
            '最高温度记录(°C)': max([day['temperature_max'] for day in extreme_days]) if extreme_days else 'N/A',
            '最低温度记录(°C)': min([day['temperature_min'] for day in extreme_days]) if extreme_days else 'N/A',
            '最大风速记录(m/s)': max([day['wind_speed_max'] for day in extreme_days]) if extreme_days else 'N/A',
            '最大降水记录(mm)': max([day['precipitation_sum'] for day in extreme_days]) if extreme_days else 'N/A'
        })
    
    return pd.DataFrame(extreme_summary)

def generate_monthly_scenario_distribution(all_results):
    """生成月度场景分布统计"""
    monthly_data = []
    
    for province, data in all_results.items():
        typical_days = data.get('typical_days', {})
        
        for scenario_name, scenario_data in typical_days.items():
            typical_dates = scenario_data['typical_dates']
            
            # 按月份统计
            monthly_counts = {}
            for date_str in typical_dates:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    month = date_obj.month
                    monthly_counts[month] = monthly_counts.get(month, 0) + 1
                except:
                    continue
            
            for month, count in monthly_counts.items():
                monthly_data.append({
                    '省份': province,
                    '场景类型': scenario_name,
                    '月份': month,
                    '月份名称': f"{month}月",
                    '典型日数量': count
                })
    
    return pd.DataFrame(monthly_data)

def save_summary_reports(typical_summary, extreme_summary, monthly_summary):
    """保存汇总报告"""
    output_dir = "results/weather_scenario_analysis/2024"
    
    # 保存Excel文件
    with pd.ExcelWriter(f"{output_dir}/2024年华东地区天气场景汇总.xlsx", engine='openpyxl') as writer:
        typical_summary.to_excel(writer, sheet_name='典型场景日汇总', index=False)
        extreme_summary.to_excel(writer, sheet_name='极端天气汇总', index=False)
        monthly_summary.to_excel(writer, sheet_name='月度分布统计', index=False)
    
    # 保存CSV文件
    typical_summary.to_csv(f"{output_dir}/典型场景日汇总.csv", index=False, encoding='utf-8-sig')
    extreme_summary.to_csv(f"{output_dir}/极端天气汇总.csv", index=False, encoding='utf-8-sig')
    monthly_summary.to_csv(f"{output_dir}/月度分布统计.csv", index=False, encoding='utf-8-sig')
    
    # 生成Markdown格式的汇总报告
    generate_markdown_summary(typical_summary, extreme_summary, monthly_summary, output_dir)

def generate_markdown_summary(typical_summary, extreme_summary, monthly_summary, output_dir):
    """生成Markdown格式的详细汇总报告"""
    
    with open(f"{output_dir}/2024年典型天气场景详细汇总.md", 'w', encoding='utf-8') as f:
        f.write("# 2024年华东地区典型天气场景详细汇总报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 典型场景日汇总
        f.write("## 一、典型天气场景日汇总\n\n")
        f.write("### 各省典型场景日一览表\n\n")
        
        # 按省份分组显示
        for province in typical_summary['省份'].unique():
            province_data = typical_summary[typical_summary['省份'] == province]
            f.write(f"#### {province}\n\n")
            
            for _, row in province_data.iterrows():
                f.write(f"**{row['场景类型']}** (出现{row['出现天数']}天，占{row['占比(%)']}%)\n")
                f.write(f"- 典型日期: {row['典型日期']}\n")
                f.write(f"- 温度特征: 平均{row['平均温度(°C)']}°C，最高{row['最高温度(°C)']}°C，最低{row['最低温度(°C)']}°C\n")
                f.write(f"- 其他特征: 湿度{row['平均湿度(%)']}%，风速{row['平均风速(m/s)']}m/s，降水{row['日降水量(mm)']}mm\n\n")
        
        # 极端天气汇总
        f.write("## 二、极端天气事件汇总\n\n")
        f.write("| 省份 | 极端天气总天数 | 主要极端天气类型 | 最严重极端天气日 | 最严重天气条件 |\n")
        f.write("|------|---------------|-----------------|-----------------|----------------|\n")
        
        for _, row in extreme_summary.iterrows():
            f.write(f"| {row['省份']} | {row['极端天气总天数']} | {row['主要极端天气类型']} | {row['最严重极端天气日']} | {row['最严重天气条件']} |\n")
        
        f.write("\n### 极端天气记录\n\n")
        f.write("| 省份 | 最高温度记录 | 最低温度记录 | 最大风速记录 | 最大降水记录 |\n")
        f.write("|------|-------------|-------------|-------------|-------------|\n")
        
        for _, row in extreme_summary.iterrows():
            f.write(f"| {row['省份']} | {row['最高温度记录(°C)']}°C | {row['最低温度记录(°C)']}°C | {row['最大风速记录(m/s)']}m/s | {row['最大降水记录(mm)']}mm |\n")
        
        # 季节性分析
        f.write("\n## 三、季节性特征分析\n\n")
        
        # 按季节统计典型场景
        seasonal_stats = {}
        for _, row in monthly_summary.iterrows():
            month = row['月份']
            if 3 <= month <= 5:
                season = '春季'
            elif 6 <= month <= 8:
                season = '夏季'
            elif 9 <= month <= 11:
                season = '秋季'
            else:
                season = '冬季'
            
            if season not in seasonal_stats:
                seasonal_stats[season] = {}
            
            scenario = row['场景类型']
            if scenario not in seasonal_stats[season]:
                seasonal_stats[season][scenario] = 0
            seasonal_stats[season][scenario] += row['典型日数量']
        
        for season, scenarios in seasonal_stats.items():
            f.write(f"### {season}\n")
            total_days = sum(scenarios.values())
            for scenario, days in sorted(scenarios.items(), key=lambda x: x[1], reverse=True):
                percentage = days / total_days * 100 if total_days > 0 else 0
                f.write(f"- {scenario}: {days}个典型日 ({percentage:.1f}%)\n")
            f.write("\n")
        
        # 电力系统运行建议
        f.write("## 四、电力系统运行建议\n\n")
        
        f.write("### 基于典型场景的调度策略\n\n")
        
        # 分析各省主要场景
        province_scenarios = {}
        for _, row in typical_summary.iterrows():
            province = row['省份']
            scenario = row['场景类型']
            count = int(row['出现天数'])
            
            if province not in province_scenarios:
                province_scenarios[province] = {}
            province_scenarios[province][scenario] = count
        
        f.write("#### 各省重点关注场景\n\n")
        for province, scenarios in province_scenarios.items():
            main_scenario = max(scenarios.items(), key=lambda x: x[1])
            f.write(f"**{province}**: 主要关注{main_scenario[0]}（{main_scenario[1]}天），")
            
            if '极端高温' in scenarios:
                f.write("夏季需重点防范高温负荷冲击；")
            if '极端低温' in scenarios:
                f.write("冬季需关注采暖负荷增长；")
            if '大风天气' in scenarios:
                f.write("需加强风电出力预测和设备防护；")
            if '暴雨天气' in scenarios:
                f.write("需防范降水对设备和新能源的影响。")
            
            f.write("\n\n")
        
        f.write("### 预警机制建议\n\n")
        f.write("1. **高温预警**: 当预测最高温度超过35°C时，启动高温负荷预警\n")
        f.write("2. **低温预警**: 当预测最低温度低于0°C时，启动低温采暖预警\n")
        f.write("3. **大风预警**: 当预测风速超过10m/s时，启动新能源出力波动预警\n")
        f.write("4. **暴雨预警**: 当预测降水量超过25mm时，启动设备防护预警\n\n")
        
        f.write("### 应急响应措施\n\n")
        f.write("- **极端高温**: 增加30%备用容量，启动需求响应，加强设备冷却\n")
        f.write("- **极端低温**: 增加20%备用容量，确保燃料供应，防范设备结冰\n")
        f.write("- **大风天气**: 加强风电功率预测，准备快速调节资源\n")
        f.write("- **暴雨天气**: 加强设备巡检，准备抢修队伍，关注新能源出力\n\n")

def main():
    """主函数"""
    print("=" * 60)
    print("生成2024年典型天气场景汇总报告")
    print("=" * 60)
    
    # 加载分析结果
    print("正在加载分析结果...")
    all_results = load_analysis_results()
    
    if not all_results:
        print("未找到分析结果文件，请先运行天气场景聚类分析脚本")
        return
    
    print(f"已加载{len(all_results)}个省份的分析结果")
    
    # 生成各类汇总表
    print("正在生成典型场景日汇总...")
    typical_summary = generate_typical_days_summary(all_results)
    
    print("正在生成极端天气汇总...")
    extreme_summary = generate_extreme_weather_summary(all_results)
    
    print("正在生成月度分布统计...")
    monthly_summary = generate_monthly_scenario_distribution(all_results)
    
    # 保存报告
    print("正在保存汇总报告...")
    save_summary_reports(typical_summary, extreme_summary, monthly_summary)
    
    print("\n" + "=" * 60)
    print("汇总报告生成完成")
    print("=" * 60)
    print(f"典型场景日总数: {len(typical_summary)}")
    print(f"极端天气事件: {extreme_summary['极端天气总天数'].sum()}天")
    print(f"覆盖省份: {len(all_results)}个")
    print("\n文件保存位置:")
    print("- Excel汇总: results/weather_scenario_analysis/2024/2024年华东地区天气场景汇总.xlsx")
    print("- 详细报告: results/weather_scenario_analysis/2024/2024年典型天气场景详细汇总.md")
    print("- CSV文件: results/weather_scenario_analysis/2024/")

if __name__ == "__main__":
    main() 