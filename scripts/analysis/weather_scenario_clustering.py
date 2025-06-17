#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全年天气场景聚类分析脚本
识别典型天气场景，分析各省的特征
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.weather_scenario_classifier import WeatherScenarioClassifier
import utils.plot_style
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 应用绘图样式
utils.plot_style.apply_style(force=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class WeatherScenarioClusteringAnalyzer:
    """天气场景聚类分析器"""
    
    def __init__(self, year=2024):
        self.year = year
        self.weather_classifier = WeatherScenarioClassifier()
        self.provinces = ['上海', '江苏', '浙江', '安徽', '福建']
        self.scenario_features = ['temperature', 'humidity', 'wind_speed', 'solar_radiation', 'precipitation']
        self.scaler = StandardScaler()
        
        # 创建输出目录
        self.output_dir = f"results/weather_scenario_analysis/{year}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_weather_data(self, province):
        """加载指定省份的天气数据"""
        try:
            # 加载真实的负荷天气数据
            weather_file = f"data/timeseries_load_weather_{province}.csv"
            if os.path.exists(weather_file):
                logger.info(f"加载真实负荷天气数据: {weather_file}")
                df = pd.read_csv(weather_file, parse_dates=['datetime'])
                
                # 重命名列以匹配原有的处理逻辑
                df = df.rename(columns={
                    'weather_temperature_c': 'temperature',
                    'weather_wind_speed': 'wind_speed',
                    'weather_relative_humidity': 'humidity',
                    'weather_precipitation_mm': 'precipitation'
                })
                
                # 添加太阳辐射估算（基于时间和天气条件）
                df['solar_radiation'] = self._estimate_solar_radiation(df)
                
                # 添加省份信息
                df['province'] = province
                
                logger.info(f"成功加载{len(df)}条{province}的真实数据记录")
                return df
            
            # 如果没有真实数据，生成全年模拟数据
            logger.warning(f"未找到{province}的真实天气数据，生成模拟数据")
            return self.generate_annual_weather_data(province)
            
        except Exception as e:
            logger.error(f"加载{province}天气数据失败: {e}")
            return self.generate_annual_weather_data(province)
    
    def _estimate_solar_radiation(self, df):
        """基于时间和天气条件估算太阳辐射"""
        solar_radiation = []
        
        for _, row in df.iterrows():
            dt = row['datetime']
            hour = dt.hour
            day_of_year = dt.timetuple().tm_yday
            
            # 基础太阳辐射（基于时间）
            if 6 <= hour <= 18:
                # 季节性变化
                seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                # 时间变化
                hourly_factor = np.sin(np.pi * (hour - 6) / 12)
                # 基础辐射
                base_radiation = 800 * seasonal_factor * hourly_factor
                
                # 天气条件修正
                # 温度影响（高温通常意味着晴朗）
                temp_factor = 1.0 + (row['temperature'] - 20) * 0.01
                # 湿度影响（高湿度通常意味着多云）
                humidity_factor = 1.0 - (row['humidity'] - 50) * 0.005
                # 降水影响
                precip_factor = 1.0 - min(row['precipitation'] * 0.1, 0.8)
                
                radiation = base_radiation * temp_factor * humidity_factor * precip_factor
                solar_radiation.append(max(0, radiation))
            else:
                solar_radiation.append(0)
        
        return solar_radiation
    
    def generate_annual_weather_data(self, province):
        """生成全年模拟天气数据"""
        start_date = datetime(self.year, 1, 1)
        end_date = datetime(self.year, 12, 31, 23, 45)
        
        # 生成时间序列（15分钟间隔）
        date_range = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        weather_data = []
        
        for dt in date_range:
            # 基于日期和时间生成模拟天气数据
            day_of_year = dt.timetuple().tm_yday
            hour = dt.hour
            
            # 季节性温度变化
            seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            daily_temp_variation = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
            random_temp_variation = np.random.normal(0, 3)
            temperature = seasonal_temp + daily_temp_variation + random_temp_variation
            
            # 添加极端天气事件
            if np.random.random() < 0.02:  # 2%概率极端高温
                temperature += np.random.uniform(10, 20)
            elif np.random.random() < 0.02:  # 2%概率极端低温
                temperature -= np.random.uniform(10, 15)
            
            # 湿度（与温度负相关）
            base_humidity = 70 - (temperature - 15) * 0.8
            humidity = max(20, min(95, base_humidity + np.random.normal(0, 10)))
            
            # 风速（季节性变化）
            seasonal_wind = 3 + 2 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
            wind_speed = max(0.5, seasonal_wind + np.random.normal(0, 2))
            
            # 添加大风天气
            if np.random.random() < 0.05:  # 5%概率大风
                wind_speed += np.random.uniform(5, 15)
            
            # 太阳辐射（季节和时间相关）
            if 6 <= hour <= 18:
                seasonal_radiation = 600 + 200 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                hourly_radiation = seasonal_radiation * np.sin(np.pi * (hour - 6) / 12)
                solar_radiation = max(0, hourly_radiation + np.random.normal(0, 50))
            else:
                solar_radiation = 0
            
            # 降水（随机事件）
            precipitation = 0
            if np.random.random() < 0.1:  # 10%概率有降水
                precipitation = np.random.exponential(2)
                # 暴雨事件
                if np.random.random() < 0.2:  # 20%概率暴雨
                    precipitation += np.random.uniform(10, 30)
            
            weather_data.append({
                'datetime': dt,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'solar_radiation': solar_radiation,
                'precipitation': precipitation,
                'province': province
            })
        
        return pd.DataFrame(weather_data)
    
    def extract_daily_features(self, weather_df):
        """提取每日天气和负荷特征"""
        weather_df['date'] = weather_df['datetime'].dt.date
        
        # 基础天气特征聚合
        daily_features = weather_df.groupby('date').agg({
            'temperature': ['mean', 'max', 'min', 'std'],
            'humidity': ['mean', 'max', 'min'],
            'wind_speed': ['mean', 'max'],
            'solar_radiation': ['mean', 'max'],
            'precipitation': ['sum', 'max'],
            'load': ['mean', 'max', 'min', 'std']  # 添加负荷特征
        }).round(2)
        
        # 扁平化列名
        daily_features.columns = ['_'.join(col).strip() for col in daily_features.columns]
        
        # 添加衍生特征
        daily_features['temp_range'] = daily_features['temperature_max'] - daily_features['temperature_min']
        daily_features['load_range'] = daily_features['load_max'] - daily_features['load_min']
        daily_features['load_volatility'] = daily_features['load_std'] / daily_features['load_mean']  # 负荷波动率
        
        # 天气条件标识
        daily_features['is_rainy_day'] = (daily_features['precipitation_sum'] > 1).astype(int)
        daily_features['is_windy_day'] = (daily_features['wind_speed_max'] > 8).astype(int)
        daily_features['is_hot_day'] = (daily_features['temperature_max'] > 35).astype(int)
        daily_features['is_cold_day'] = (daily_features['temperature_min'] < 0).astype(int)
        
        # 负荷水平标识
        load_mean_overall = daily_features['load_mean'].mean()
        daily_features['is_high_load_day'] = (daily_features['load_mean'] > load_mean_overall * 1.2).astype(int)
        daily_features['is_low_load_day'] = (daily_features['load_mean'] < load_mean_overall * 0.8).astype(int)
        
        return daily_features.reset_index()
    
    def identify_extreme_weather_days(self, daily_features):
        """识别极端天气日"""
        extreme_days = []
        
        for idx, row in daily_features.iterrows():
            date = row['date']
            conditions = []
            
            # 极端高温
            if row['temperature_max'] > 38:
                conditions.append('极端高温')
            
            # 极端低温
            if row['temperature_min'] < -5:
                conditions.append('极端低温')
            
            # 大风
            if row['wind_speed_max'] > 12:
                conditions.append('大风')
            
            # 暴雨
            if row['precipitation_sum'] > 25:
                conditions.append('暴雨')
            
            # 高温高湿
            if row['temperature_max'] > 32 and row['humidity_mean'] > 80:
                conditions.append('高温高湿')
            
            # 低温大风
            if row['temperature_mean'] < 5 and row['wind_speed_max'] > 8:
                conditions.append('低温大风')
            
            if conditions:
                extreme_days.append({
                    'date': date,
                    'conditions': conditions,
                    'temperature_max': row['temperature_max'],
                    'temperature_min': row['temperature_min'],
                    'wind_speed_max': row['wind_speed_max'],
                    'precipitation_sum': row['precipitation_sum'],
                    'humidity_mean': row['humidity_mean']
                })
        
        return pd.DataFrame(extreme_days)
    
    def perform_weather_clustering(self, daily_features, n_clusters=8):
        """执行天气和负荷聚类分析"""
        # 选择聚类特征（包含天气和负荷特征）
        feature_columns = [
            'temperature_mean', 'temperature_max', 'temperature_min', 'temp_range',
            'humidity_mean', 'wind_speed_mean', 'wind_speed_max',
            'solar_radiation_mean', 'precipitation_sum',
            'load_mean', 'load_max', 'load_range', 'load_volatility'  # 添加负荷特征
        ]
        
        X = daily_features[feature_columns].fillna(0)
        
        # 处理可能的无穷值
        X = X.replace([np.inf, -np.inf], 0)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 寻找最优聚类数
        silhouette_scores = []
        K_range = range(3, 12)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # 选择最优K值
        optimal_k = K_range[np.argmax(silhouette_scores)]
        logger.info(f"最优聚类数: {optimal_k}, 轮廓系数: {max(silhouette_scores):.3f}")
        
        # 执行最终聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        daily_features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # PCA降维用于可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        daily_features['pca1'] = X_pca[:, 0]
        daily_features['pca2'] = X_pca[:, 1]
        
        return daily_features, kmeans, feature_columns, optimal_k
    
    def analyze_cluster_characteristics(self, daily_features, feature_columns):
        """分析聚类特征（包含天气和负荷特征）"""
        cluster_stats = daily_features.groupby('cluster')[feature_columns].agg(['mean', 'std']).round(2)
        
        # 为每个聚类命名
        cluster_names = {}
        cluster_descriptions = {}
        
        for cluster_id in daily_features['cluster'].unique():
            cluster_data = daily_features[daily_features['cluster'] == cluster_id]
            
            # 分析聚类特征
            temp_mean = cluster_data['temperature_mean'].mean()
            temp_max = cluster_data['temperature_max'].mean()
            wind_max = cluster_data['wind_speed_max'].mean()
            precip_sum = cluster_data['precipitation_sum'].mean()
            humidity_mean = cluster_data['humidity_mean'].mean()
            load_mean = cluster_data['load_mean'].mean()
            load_volatility = cluster_data['load_volatility'].mean()
            
            # 计算整体负荷水平参考
            overall_load_mean = daily_features['load_mean'].mean()
            
            # 根据天气和负荷特征命名聚类
            name_parts = []
            desc_parts = []
            
            # 温度特征
            if temp_max > 35:
                name_parts.append("极端高温")
                desc_parts.append(f"最高温度{temp_max:.1f}°C")
            elif temp_mean < 5:
                name_parts.append("极端低温")
                desc_parts.append(f"平均温度{temp_mean:.1f}°C")
            elif 25 <= temp_mean <= 30:
                name_parts.append("温和")
                desc_parts.append(f"温度适宜{temp_mean:.1f}°C")
            
            # 其他天气特征
            if wind_max > 10:
                name_parts.append("大风")
                desc_parts.append(f"风速{wind_max:.1f}m/s")
            if precip_sum > 15:
                name_parts.append("多雨")
                desc_parts.append(f"降水{precip_sum:.1f}mm")
            if temp_max > 30 and humidity_mean > 75:
                name_parts.append("高湿")
                desc_parts.append(f"湿度{humidity_mean:.1f}%")
            
            # 负荷特征
            if load_mean > overall_load_mean * 1.15:
                name_parts.append("高负荷")
                desc_parts.append(f"负荷{load_mean:.0f}MW")
            elif load_mean < overall_load_mean * 0.85:
                name_parts.append("低负荷")
                desc_parts.append(f"负荷{load_mean:.0f}MW")
            
            if load_volatility > 0.15:
                name_parts.append("高波动")
                desc_parts.append(f"波动率{load_volatility:.2f}")
            
            # 组合命名
            if name_parts:
                name = "".join(name_parts[:3])  # 最多取3个特征
            else:
                name = f"一般场景{cluster_id}"
            
            desc = "，".join(desc_parts) if desc_parts else f"一般天气和负荷条件"
            
            cluster_names[cluster_id] = name
            cluster_descriptions[cluster_id] = desc
        
        return cluster_stats, cluster_names, cluster_descriptions
    
    def identify_typical_scenario_days(self, daily_features, cluster_names):
        """识别典型场景日（包含天气和负荷特征）"""
        typical_days = {}
        
        for cluster_id, cluster_name in cluster_names.items():
            cluster_data = daily_features[daily_features['cluster'] == cluster_id].copy()
            
            if len(cluster_data) == 0:
                continue
            
            # 计算每天到聚类中心的距离（包含天气和负荷特征）
            feature_columns = [
                'temperature_mean', 'temperature_max', 'temperature_min',
                'humidity_mean', 'wind_speed_mean', 'precipitation_sum',
                'load_mean', 'load_max', 'load_volatility'  # 添加负荷特征
            ]
            
            cluster_center = cluster_data[feature_columns].mean()
            
            distances = []
            for idx, row in cluster_data.iterrows():
                distance = np.sqrt(sum((row[col] - cluster_center[col])**2 for col in feature_columns))
                distances.append(distance)
            
            cluster_data['distance_to_center'] = distances
            
            # 选择最接近聚类中心的几天作为典型日
            typical_days_data = cluster_data.nsmallest(3, 'distance_to_center')
            
            typical_days[cluster_name] = {
                'cluster_id': cluster_id,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(daily_features) * 100,
                'typical_dates': typical_days_data['date'].tolist(),
                'characteristics': {
                    'temperature_mean': cluster_center['temperature_mean'],
                    'temperature_max': cluster_center['temperature_max'],
                    'temperature_min': cluster_center['temperature_min'],
                    'humidity_mean': cluster_center['humidity_mean'],
                    'wind_speed_mean': cluster_center['wind_speed_mean'],
                    'precipitation_sum': cluster_center['precipitation_sum'],
                    'load_mean': cluster_center['load_mean'],
                    'load_max': cluster_center['load_max'],
                    'load_volatility': cluster_center['load_volatility']
                }
            }
        
        return typical_days
    
    def create_visualizations(self, daily_features, cluster_names, province):
        """创建可视化图表（包含负荷分析）"""
        # 应用绘图样式
        utils.plot_style.apply_style(force=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 聚类散点图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{province} {self.year}年天气负荷场景聚类分析', fontsize=16, fontweight='bold')
        
        # PCA散点图
        scatter = axes[0, 0].scatter(daily_features['pca1'], daily_features['pca2'], 
                                   c=daily_features['cluster'], cmap='tab10', alpha=0.7)
        axes[0, 0].set_title('PCA降维聚类结果')
        axes[0, 0].set_xlabel('第一主成分')
        axes[0, 0].set_ylabel('第二主成分')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 温度-负荷散点图
        scatter2 = axes[0, 1].scatter(daily_features['temperature_mean'], daily_features['load_mean'],
                                    c=daily_features['cluster'], cmap='tab10', alpha=0.7)
        axes[0, 1].set_title('温度-负荷关系')
        axes[0, 1].set_xlabel('平均温度 (°C)')
        axes[0, 1].set_ylabel('平均负荷 (MW)')
        
        # 聚类分布饼图
        cluster_counts = daily_features['cluster'].value_counts().sort_index()
        cluster_labels = [cluster_names.get(i, f'聚类{i}') for i in cluster_counts.index]
        axes[1, 0].pie(cluster_counts.values, labels=cluster_labels, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('天气负荷场景分布')
        
        # 月度聚类分布
        daily_features['month'] = pd.to_datetime(daily_features['date']).dt.month
        monthly_cluster = daily_features.groupby(['month', 'cluster']).size().unstack(fill_value=0)
        monthly_cluster.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='tab10')
        axes[1, 1].set_title('月度场景分布')
        axes[1, 1].set_xlabel('月份')
        axes[1, 1].set_ylabel('天数')
        axes[1, 1].legend(title='聚类', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{province}_weather_load_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 特征分布箱线图
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'{province} {self.year}年各聚类天气负荷特征分布', fontsize=16, fontweight='bold')
        
        features_to_plot = [
            ('temperature_mean', '平均温度 (°C)'),
            ('humidity_mean', '平均湿度 (%)'),
            ('wind_speed_mean', '平均风速 (m/s)'),
            ('solar_radiation_mean', '平均太阳辐射 (W/m²)'),
            ('precipitation_sum', '日降水量 (mm)'),
            ('temp_range', '日温差 (°C)'),
            ('load_mean', '平均负荷 (MW)'),
            ('load_range', '负荷日变化 (MW)'),
            ('load_volatility', '负荷波动率')
        ]
        
        for idx, (feature, title) in enumerate(features_to_plot):
            row, col = idx // 3, idx % 3
            
            # 准备数据
            data_for_box = []
            labels_for_box = []
            
            for cluster_id in sorted(daily_features['cluster'].unique()):
                cluster_data = daily_features[daily_features['cluster'] == cluster_id][feature]
                data_for_box.append(cluster_data)
                labels_for_box.append(cluster_names.get(cluster_id, f'聚类{cluster_id}'))
            
            axes[row, col].boxplot(data_for_box, labels=labels_for_box)
            axes[row, col].set_title(title)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{province}_feature_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 负荷与天气关系分析图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{province} {self.year}年负荷与天气关系分析', fontsize=16, fontweight='bold')
        
        # 温度vs负荷
        axes[0, 0].scatter(daily_features['temperature_mean'], daily_features['load_mean'], 
                          c=daily_features['cluster'], cmap='tab10', alpha=0.6)
        axes[0, 0].set_xlabel('平均温度 (°C)')
        axes[0, 0].set_ylabel('平均负荷 (MW)')
        axes[0, 0].set_title('温度-负荷关系')
        
        # 湿度vs负荷
        axes[0, 1].scatter(daily_features['humidity_mean'], daily_features['load_mean'], 
                          c=daily_features['cluster'], cmap='tab10', alpha=0.6)
        axes[0, 1].set_xlabel('平均湿度 (%)')
        axes[0, 1].set_ylabel('平均负荷 (MW)')
        axes[0, 1].set_title('湿度-负荷关系')
        
        # 月度负荷变化
        monthly_load = daily_features.groupby('month')['load_mean'].mean()
        axes[1, 0].plot(monthly_load.index, monthly_load.values, marker='o', linewidth=2)
        axes[1, 0].set_xlabel('月份')
        axes[1, 0].set_ylabel('平均负荷 (MW)')
        axes[1, 0].set_title('月度负荷变化趋势')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 负荷波动率分布
        axes[1, 1].hist(daily_features['load_volatility'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('负荷波动率')
        axes[1, 1].set_ylabel('天数')
        axes[1, 1].set_title('负荷波动率分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{province}_load_weather_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, province, typical_days, extreme_days, cluster_descriptions):
        """生成分析报告（包含负荷分析）"""
        report_file = f'{self.output_dir}/{province}_weather_load_scenario_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# {province} {self.year}年天气负荷场景聚类分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 概述
            f.write("## 分析概述\n\n")
            f.write(f"本报告基于{self.year}年全年真实负荷和天气数据，采用K-means聚类算法识别{province}的典型天气负荷场景。\n")
            f.write(f"共识别出{len(typical_days)}种主要场景类型，综合考虑了温度、湿度、风速、降水等天气因素以及负荷水平、波动性等电力特征。\n\n")
            
            # 典型场景分析
            f.write("## 典型天气负荷场景\n\n")
            for scenario_name, scenario_data in typical_days.items():
                f.write(f"### {scenario_name}\n\n")
                f.write(f"- **出现天数**: {scenario_data['count']}天 ({scenario_data['percentage']:.1f}%)\n")
                f.write(f"- **典型日期**: {', '.join([str(d) for d in scenario_data['typical_dates'][:3]])}\n")
                
                chars = scenario_data['characteristics']
                f.write(f"- **天气特征**:\n")
                f.write(f"  - 平均温度: {chars['temperature_mean']:.1f}°C\n")
                f.write(f"  - 最高温度: {chars['temperature_max']:.1f}°C\n")
                f.write(f"  - 最低温度: {chars['temperature_min']:.1f}°C\n")
                f.write(f"  - 平均湿度: {chars['humidity_mean']:.1f}%\n")
                f.write(f"  - 平均风速: {chars['wind_speed_mean']:.1f}m/s\n")
                f.write(f"  - 日降水量: {chars['precipitation_sum']:.1f}mm\n")
                
                f.write(f"- **负荷特征**:\n")
                f.write(f"  - 平均负荷: {chars['load_mean']:.0f}MW\n")
                f.write(f"  - 最大负荷: {chars['load_max']:.0f}MW\n")
                f.write(f"  - 负荷波动率: {chars['load_volatility']:.3f}\n\n")
            
            # 极端天气事件
            if not extreme_days.empty:
                f.write("## 极端天气事件\n\n")
                f.write(f"全年共识别出{len(extreme_days)}个极端天气日。\n\n")
                
                # 按条件分类统计
                condition_counts = {}
                for _, row in extreme_days.iterrows():
                    for condition in row['conditions']:
                        condition_counts[condition] = condition_counts.get(condition, 0) + 1
                
                f.write("### 极端天气类型统计\n\n")
                for condition, count in sorted(condition_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {condition}: {count}天\n")
                
                f.write("\n### 典型极端天气日\n\n")
                for _, row in extreme_days.head(10).iterrows():
                    f.write(f"- **{row['date']}**: {', '.join(row['conditions'])}\n")
                    f.write(f"  - 最高温度: {row['temperature_max']:.1f}°C\n")
                    f.write(f"  - 最低温度: {row['temperature_min']:.1f}°C\n")
                    f.write(f"  - 最大风速: {row['wind_speed_max']:.1f}m/s\n")
                    f.write(f"  - 降水量: {row['precipitation_sum']:.1f}mm\n\n")
            
            # 负荷特征分析
            f.write("## 负荷特征分析\n\n")
            
            # 计算负荷统计信息
            all_load_data = []
            for scenario_name, scenario_data in typical_days.items():
                chars = scenario_data['characteristics']
                all_load_data.append({
                    'scenario': scenario_name,
                    'load_mean': chars['load_mean'],
                    'load_max': chars['load_max'],
                    'volatility': chars['load_volatility'],
                    'count': scenario_data['count']
                })
            
            load_df = pd.DataFrame(all_load_data)
            
            f.write("### 各场景负荷水平对比\n\n")
            f.write("| 场景类型 | 平均负荷(MW) | 最大负荷(MW) | 波动率 | 出现天数 |\n")
            f.write("|----------|-------------|-------------|--------|----------|\n")
            
            for _, row in load_df.iterrows():
                f.write(f"| {row['scenario']} | {row['load_mean']:.0f} | {row['load_max']:.0f} | {row['volatility']:.3f} | {row['count']} |\n")
            
            # 季节性分析
            f.write("\n## 季节性特征\n\n")
            f.write("各季节主要天气负荷场景分布:\n\n")
            
            f.write("- **春季(3-5月)**: 温和气候，负荷相对稳定，适合设备检修\n")
            f.write("- **夏季(6-8月)**: 高温高负荷，空调负荷占主导，系统压力大\n")
            f.write("- **秋季(9-11月)**: 秋高气爽，负荷适中，系统运行平稳\n")
            f.write("- **冬季(12-2月)**: 低温采暖，负荷波动较大，需关注燃料供应\n\n")
            
            # 电力系统运行建议
            f.write("## 电力系统运行建议\n\n")
            f.write("基于天气负荷场景分析，对电力系统运行提出以下建议:\n\n")
            
            # 根据场景特征给出具体建议
            high_load_scenarios = [s for s in typical_days.keys() if '高负荷' in s or '极端高温' in s]
            high_volatility_scenarios = [s for s in typical_days.keys() if '高波动' in s]
            
            if high_load_scenarios:
                f.write(f"1. **高负荷场景({', '.join(high_load_scenarios)})**: \n")
                f.write("   - 增加30-50%备用容量\n")
                f.write("   - 启动需求响应机制\n")
                f.write("   - 加强设备监控和维护\n")
                f.write("   - 优化机组组合和调度策略\n\n")
            
            if high_volatility_scenarios:
                f.write(f"2. **高波动场景({', '.join(high_volatility_scenarios)})**: \n")
                f.write("   - 增加快速调节资源\n")
                f.write("   - 加强负荷预测精度\n")
                f.write("   - 准备应急调度预案\n")
                f.write("   - 协调储能系统参与调节\n\n")
            
            f.write("3. **极端天气应对**: \n")
            f.write("   - 建立天气负荷场景预警机制\n")
            f.write("   - 制定分场景的运行策略\n")
            f.write("   - 加强与气象部门的信息共享\n")
            f.write("   - 提升新能源出力预测精度\n\n")
            
            f.write("4. **日常运行优化**: \n")
            f.write("   - 基于历史场景优化调度计划\n")
            f.write("   - 建立场景化的备用容量配置\n")
            f.write("   - 完善负荷预测模型\n")
            f.write("   - 加强设备状态监测\n\n")
        
        logger.info(f"分析报告已保存至: {report_file}")
    
    def analyze_province(self, province):
        """分析单个省份的天气场景"""
        logger.info(f"开始分析{province}的天气场景...")
        
        # 1. 加载天气数据
        weather_df = self.load_weather_data(province)
        
        # 2. 提取每日特征
        daily_features = self.extract_daily_features(weather_df)
        
        # 3. 识别极端天气日
        extreme_days = self.identify_extreme_weather_days(daily_features)
        
        # 4. 执行聚类分析
        daily_features, kmeans, feature_columns, optimal_k = self.perform_weather_clustering(daily_features)
        
        # 5. 分析聚类特征
        cluster_stats, cluster_names, cluster_descriptions = self.analyze_cluster_characteristics(
            daily_features, feature_columns)
        
        # 6. 识别典型场景日
        typical_days = self.identify_typical_scenario_days(daily_features, cluster_names)
        
        # 7. 创建可视化
        self.create_visualizations(daily_features, cluster_names, province)
        
        # 8. 生成报告
        self.generate_report(province, typical_days, extreme_days, cluster_descriptions)
        
        # 9. 保存结果数据
        results = {
            'province': province,
            'year': self.year,
            'typical_days': typical_days,
            'extreme_days': extreme_days.to_dict('records') if not extreme_days.empty else [],
            'cluster_descriptions': cluster_descriptions,
            'total_days': len(daily_features),
            'analysis_date': datetime.now().isoformat()
        }
        
        # 保存为JSON
        import json
        
        # 转换numpy数据类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_converted = convert_numpy_types(results)
        
        with open(f'{self.output_dir}/{province}_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"{province}分析完成，结果保存至 {self.output_dir}")
        return results
    
    def analyze_all_provinces(self):
        """分析所有省份的天气场景"""
        all_results = {}
        
        for province in self.provinces:
            try:
                results = self.analyze_province(province)
                all_results[province] = results
            except Exception as e:
                logger.error(f"分析{province}时出错: {e}")
                continue
        
        # 生成综合报告
        self.generate_comprehensive_report(all_results)
        
        return all_results
    
    def generate_comprehensive_report(self, all_results):
        """生成综合分析报告"""
        report_file = f'{self.output_dir}/comprehensive_weather_analysis_{self.year}.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# {self.year}年华东地区天气场景综合分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 概述
            f.write("## 分析概述\n\n")
            f.write(f"本报告分析了华东地区{len(all_results)}个省份在{self.year}年的天气场景特征。\n")
            f.write("采用机器学习聚类算法识别典型天气模式，为电力系统运行提供决策支持。\n\n")
            
            # 各省典型场景对比
            f.write("## 各省典型天气场景对比\n\n")
            f.write("| 省份 | 主要场景类型 | 极端天气日数 | 特色天气特征 |\n")
            f.write("|------|-------------|-------------|-------------|\n")
            
            for province, results in all_results.items():
                typical_days = results['typical_days']
                extreme_days_count = len(results['extreme_days'])
                
                # 找出最主要的场景
                main_scenarios = sorted(typical_days.items(), 
                                      key=lambda x: x[1]['count'], reverse=True)[:3]
                main_scenario_names = [s[0] for s in main_scenarios]
                
                f.write(f"| {province} | {', '.join(main_scenario_names)} | {extreme_days_count} | ")
                
                # 特色特征
                if extreme_days_count > 50:
                    f.write("极端天气频发")
                elif any('高温' in name for name in main_scenario_names):
                    f.write("夏季高温显著")
                elif any('低温' in name for name in main_scenario_names):
                    f.write("冬季严寒")
                else:
                    f.write("气候温和")
                
                f.write(" |\n")
            
            # 区域性天气特征
            f.write("\n## 区域性天气特征分析\n\n")
            f.write("### 共同特征\n")
            f.write("- 夏季普遍高温高湿，空调负荷压力大\n")
            f.write("- 春秋季节气候宜人，系统运行相对稳定\n")
            f.write("- 冬季低温干燥，采暖负荷增加\n\n")
            
            f.write("### 地域差异\n")
            f.write("- 沿海地区受海洋性气候影响，温度变化相对缓和\n")
            f.write("- 内陆地区大陆性气候特征明显，温差较大\n")
            f.write("- 南部地区夏季高温持续时间更长\n")
            f.write("- 北部地区冬季低温更为严峻\n\n")
            
            # 电力系统运行建议
            f.write("## 电力系统运行建议\n\n")
            f.write("### 季节性调度策略\n")
            f.write("1. **夏季(6-8月)**:\n")
            f.write("   - 重点关注高温高湿天气的负荷预测\n")
            f.write("   - 增加备用容量应对空调负荷激增\n")
            f.write("   - 加强设备散热和维护\n\n")
            
            f.write("2. **冬季(12-2月)**:\n")
            f.write("   - 关注低温天气的采暖负荷增长\n")
            f.write("   - 做好燃料储备和供应保障\n")
            f.write("   - 防范极端低温对设备的影响\n\n")
            
            f.write("3. **春秋季(3-5月, 9-11月)**:\n")
            f.write("   - 利用温和天气优化经济调度\n")
            f.write("   - 安排设备检修和维护\n")
            f.write("   - 适当降低备用容量\n\n")
            
            f.write("### 极端天气应对\n")
            f.write("- 建立天气场景预警机制\n")
            f.write("- 制定分场景的应急预案\n")
            f.write("- 加强与气象部门的信息共享\n")
            f.write("- 提升新能源出力预测精度\n\n")
        
        logger.info(f"综合分析报告已保存至: {report_file}")

def main():
    """主函数"""
    print("=" * 60)
    print("全年天气场景聚类分析系统")
    print("=" * 60)
    
    # 创建分析器
    analyzer = WeatherScenarioClusteringAnalyzer(year=2024)
    
    # 分析所有省份
    print("\n开始分析各省份天气场景...")
    results = analyzer.analyze_all_provinces()
    
    # 输出简要结果
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)
    
    for province, result in results.items():
        print(f"\n{province}:")
        print(f"  典型场景数: {len(result['typical_days'])}")
        print(f"  极端天气日: {len(result['extreme_days'])}")
        
        # 显示主要场景
        main_scenarios = sorted(result['typical_days'].items(), 
                              key=lambda x: x[1]['count'], reverse=True)[:3]
        print(f"  主要场景: {', '.join([s[0] for s in main_scenarios])}")
    
    print(f"\n详细结果已保存至: {analyzer.output_dir}")
    print("包含可视化图表、分析报告和数据文件")

if __name__ == "__main__":
    main() 