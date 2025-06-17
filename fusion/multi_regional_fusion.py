"""
Multi-regional Net Load Forecasting with Interval Fusion
多区域净负荷区间预测融合模块

基于指标体系和主成分分析的加权融合方法，支持：
1. 多省份区间负荷预测融合
2. 新能源预测集成（光伏+风电）
3. 净负荷计算（负荷-新能源出力）
4. 不确定性传播和区间计算
5. PCA权重动态调整
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .weighted_fusion import WeightedFusion
from .pca_weights import PCAWeightCalculator
from .evaluation_index import EvaluationIndexCalculator


@dataclass
class RegionalForecastData:
    """单个区域的预测数据结构"""
    region_name: str
    load_predictions: pd.DataFrame  # 包含 predicted, lower_bound, upper_bound
    pv_predictions: np.ndarray = None
    wind_predictions: np.ndarray = None
    datetime_index: pd.DatetimeIndex = None
    actual_load: pd.DataFrame = None  # 用于评估，包含 load 列
    
    def __post_init__(self):
        """数据验证和预处理"""
        if self.datetime_index is None and hasattr(self.load_predictions, 'index'):
            self.datetime_index = self.load_predictions.index
        
        # 确保新能源预测数据长度一致
        if self.pv_predictions is not None:
            assert len(self.pv_predictions) == len(self.load_predictions), \
                f"PV predictions length {len(self.pv_predictions)} != load predictions length {len(self.load_predictions)}"
        
        if self.wind_predictions is not None:
            assert len(self.wind_predictions) == len(self.load_predictions), \
                f"Wind predictions length {len(self.wind_predictions)} != load predictions length {len(self.load_predictions)}"


class MultiRegionalNetLoadFusion:
    """
    多区域净负荷预测融合器
    
    核心功能：
    1. 集成多个区域的区间负荷预测
    2. 考虑光伏和风电出力预测
    3. 基于指标体系计算动态权重
    4. 执行区间预测的加权融合
    5. 计算最终的净负荷区间预测
    """
    
    def __init__(self, 
                 smoothing_coefficient: float = 0.7,
                 smoothing_window: int = 3,
                 max_weight_factor: float = 1.2,
                 base_adjustment_coefficient: float = 0.5):
        """
        初始化融合器
        
        Args:
            smoothing_coefficient: PCA权重时间平滑系数
            smoothing_window: 时间平滑窗口大小  
            max_weight_factor: 最大权重限制因子
            base_adjustment_coefficient: 基础调整系数
        """
        self.pca_calculator = PCAWeightCalculator(
            smoothing_coefficient=smoothing_coefficient,
            smoothing_window=smoothing_window,
            max_weight_factor=max_weight_factor
        )
        self.weighted_fusion = WeightedFusion(
            base_adjustment_coefficient=base_adjustment_coefficient
        )
        self.evaluation_calculator = EvaluationIndexCalculator()
        
        # 存储区域数据和权重历史
        self.regional_data: Dict[str, RegionalForecastData] = {}
        self.weight_history: List[Dict[str, float]] = []
        self.fusion_results = None
    
    def load_regional_data(self, regional_data: Dict[str, RegionalForecastData]):
        """
        加载区域预测数据
        
        Args:
            regional_data: 区域名称到RegionalForecastData的映射
        """
        self.regional_data = regional_data
        print(f"已加载 {len(regional_data)} 个区域的预测数据:")
        for name, data in regional_data.items():
            renewable_info = []
            if data.pv_predictions is not None:
                renewable_info.append(f"光伏({len(data.pv_predictions)}点)")
            if data.wind_predictions is not None:
                renewable_info.append(f"风电({len(data.wind_predictions)}点)")
            renewable_str = "+".join(renewable_info) if renewable_info else "无新能源"
            
            print(f"  - {name}: 负荷预测({len(data.load_predictions)}点) + {renewable_str}")
    
    def load_from_json_files(self, json_file_paths: Dict[str, str], 
                           actual_data_paths: Optional[Dict[str, str]] = None):
        """
        从JSON文件加载区域预测数据
        
        Args:
            json_file_paths: 区域名称到JSON文件路径的映射
            actual_data_paths: 区域名称到实际数据文件路径的映射（可选，用于评估）
        """
        regional_data = {}
        
        for region_name, json_path in json_file_paths.items():
            print(f"正在加载 {region_name} 的预测数据: {json_path}")
            
            # 加载JSON数据
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 解析负荷预测数据
            predictions_list = json_data['predictions']
            load_df = pd.DataFrame([
                {
                    'predicted': pred['predicted'],
                    'lower_bound': pred['lower_bound'], 
                    'upper_bound': pred['upper_bound'],
                    'datetime': pred['datetime']
                }
                for pred in predictions_list
            ])
            load_df['datetime'] = pd.to_datetime(load_df['datetime'])
            load_df.set_index('datetime', inplace=True)
            
            # 解析新能源预测数据
            pv_predictions = None
            wind_predictions = None
            
            if 'renewable_predictions' in json_data and json_data['renewable_predictions']:
                renewable_data = json_data['renewable_predictions']
                
                if 'pv' in renewable_data and renewable_data['pv']['predictions']:
                    pv_predictions = np.array(renewable_data['pv']['predictions'])
                
                if 'wind' in renewable_data and renewable_data['wind']['predictions']:
                    wind_predictions = np.array(renewable_data['wind']['predictions'])
            
            # 加载实际数据（如果提供）
            actual_load = None
            if actual_data_paths and region_name in actual_data_paths:
                try:
                    actual_df = pd.read_csv(actual_data_paths[region_name])
                    actual_df['datetime'] = pd.to_datetime(actual_df['datetime'])
                    actual_df.set_index('datetime', inplace=True)
                    actual_load = actual_df[['load']] if 'load' in actual_df.columns else None
                except Exception as e:
                    print(f"  警告: 无法加载{region_name}的实际数据: {e}")
            
            # 创建区域数据对象
            regional_data[region_name] = RegionalForecastData(
                region_name=region_name,
                load_predictions=load_df,
                pv_predictions=pv_predictions,
                wind_predictions=wind_predictions,
                datetime_index=load_df.index,
                actual_load=actual_load
            )
            
            print(f"  ✓ {region_name}: 负荷{len(load_df)}点")
            if pv_predictions is not None:
                print(f"    + 光伏{len(pv_predictions)}点")
            if wind_predictions is not None:
                print(f"    + 风电{len(wind_predictions)}点")
        
        self.load_regional_data(regional_data)
    
    def calculate_net_load_forecasts(self) -> Dict[str, pd.DataFrame]:
        """
        计算各区域的净负荷预测（负荷 - 新能源出力）
        
        Returns:
            各区域净负荷预测的字典，包含predicted, lower_bound, upper_bound列
        """
        net_load_forecasts = {}
        
        for region_name, data in self.regional_data.items():
            # 复制原始负荷预测
            net_load = data.load_predictions.copy()
            
            # 计算新能源总出力
            renewable_output = np.zeros(len(net_load))
            renewable_uncertainty = np.zeros(len(net_load))  # 新能源不确定性
            
            if data.pv_predictions is not None:
                renewable_output += data.pv_predictions
                # 光伏不确定性约为预测值的15%
                renewable_uncertainty += 0.15 * np.abs(data.pv_predictions)
            
            if data.wind_predictions is not None:
                renewable_output += data.wind_predictions  
                # 风电不确定性约为预测值的20%
                renewable_uncertainty += 0.20 * np.abs(data.wind_predictions)
            
            # 计算净负荷 = 负荷 - 新能源出力
            net_load['predicted'] = net_load['predicted'] - renewable_output
            
            # 不确定性传播：净负荷的不确定性 = 负荷不确定性 + 新能源不确定性
            load_uncertainty = (net_load['upper_bound'] - net_load['lower_bound']) / 2
            total_uncertainty = np.sqrt(load_uncertainty**2 + renewable_uncertainty**2)
            
            # 更新区间边界
            net_load['lower_bound'] = net_load['predicted'] - total_uncertainty
            net_load['upper_bound'] = net_load['predicted'] + total_uncertainty
            
            net_load_forecasts[region_name] = net_load
            
            print(f"{region_name} 净负荷计算完成:")
            print(f"  平均净负荷: {net_load['predicted'].mean():.1f} MW")
            print(f"  平均新能源出力: {renewable_output.mean():.1f} MW") 
            print(f"  平均区间宽度: {(net_load['upper_bound'] - net_load['lower_bound']).mean():.1f} MW")
        
        return net_load_forecasts
    
    def calculate_fusion_weights(self, net_load_forecasts: Dict[str, pd.DataFrame],
                               use_time_varying: bool = True) -> Dict[str, float]:
        """
        基于指标体系和PCA计算融合权重
        
        Args:
            net_load_forecasts: 各区域净负荷预测
            use_time_varying: 是否使用时变权重
            
        Returns:
            区域权重字典或时变权重DataFrame
        """
        print("正在计算融合权重...")
        
        # 1. 构造评估数据（如果有实际数据则用实际数据，否则用预测数据作为近似）
        actual_data = {}
        forecast_data = {}
        
        for region_name, net_load in net_load_forecasts.items():
            region_data = self.regional_data[region_name]
            
            # 准备用于评估的数据
            if region_data.actual_load is not None:
                # 有实际数据，直接使用
                actual_data[region_name] = region_data.actual_load
            else:
                # 没有实际数据，使用负荷预测值作为近似
                actual_data[region_name] = pd.DataFrame({
                    'load': net_load['predicted']
                }, index=net_load.index)
            
            # 预测数据
            forecast_data[region_name] = pd.DataFrame({
                'load': net_load['predicted']
            }, index=net_load.index)
        
        # 2. 计算评估指标
        evaluation_results = self.evaluation_calculator.evaluate_regions(
            actual_data, forecast_data
        )
        
        print("区域评估结果:")
        for region, results in evaluation_results.items():
            indices = results['indices']
            print(f"  {region}:")
            print(f"    预测可靠性: {indices['ForecastReliability']:.3f}")
            print(f"    省级影响力: {indices['ProvincialLoadImpact']:.3f}")
            print(f"    预测复杂性: {indices['ForecastingComplexity']:.3f}")
            print(f"    综合评分: {indices['FinalScore']:.3f}")
        
        # 3. 计算权重
        if use_time_varying:
            time_points = list(net_load_forecasts.values())[0].index
            weights = self.pca_calculator.calculate_time_varying_weights(
                evaluation_results, time_points
            )
            print(f"计算得到时变权重，形状: {weights.shape}")
            print("前3个时间点的权重:")
            print(weights.head(3))
        else:
            weights = self.pca_calculator.calculate_weights(evaluation_results)
            print("计算得到静态权重:")
            for region, weight in weights.items():
                print(f"  {region}: {weight:.4f}")
        
        return weights
    
    def execute_interval_fusion(self, net_load_forecasts: Dict[str, pd.DataFrame],
                              weights: Dict[str, float]) -> pd.DataFrame:
        """
        执行区间预测的加权融合 - 修正版本
        
        关键修正：
        1. 预测值使用直接求和（各省份净负荷的总和）
        2. 权重主要用于不确定性的合成，体现各省份对整体不确定性的贡献
        
        Args:
            net_load_forecasts: 各区域净负荷预测
            weights: 融合权重（主要影响不确定性合成）
            
        Returns:
            融合后的净负荷预测，包含predicted, lower_bound, upper_bound列
        """
        print("正在执行区间预测融合（修正版本）...")
        
        # 获取时间索引（以第一个区域为基准）
        reference_region = list(net_load_forecasts.keys())[0]
        time_index = net_load_forecasts[reference_region].index
        
        # 初始化融合结果
        fused_forecast = pd.DataFrame(
            index=time_index,
            columns=['predicted', 'lower_bound', 'upper_bound'],
            dtype=float
        )
        fused_forecast.fillna(0.0, inplace=True)
        
        # 检查权重类型
        is_time_varying = isinstance(weights, pd.DataFrame)
        
        # 对每个时间点进行融合
        for timestamp in time_index:
            # 获取当前时间点的权重
            if is_time_varying:
                if timestamp in weights.index:
                    current_weights = weights.loc[timestamp]
                else:
                    # 找最近的权重
                    nearest_idx = weights.index.get_indexer([timestamp], method='nearest')[0]
                    current_weights = weights.iloc[nearest_idx]
                current_weights = current_weights.to_dict()
            else:
                current_weights = weights
            
            # 1. 预测值直接求和（各省份净负荷的总和）
            total_predicted = 0.0
            
            # 2. 不确定性的加权合成
            weighted_variance = 0.0  # 加权方差合成
            total_weight = 0.0
            
            for region_name, region_forecast in net_load_forecasts.items():
                if timestamp in region_forecast.index:
                    region_weight = current_weights.get(region_name, 1.0/len(net_load_forecasts))
                    
                    # 预测值直接累加
                    total_predicted += region_forecast.loc[timestamp, 'predicted']
                    
                    # 不确定性的加权合成
                    region_uncertainty = (region_forecast.loc[timestamp, 'upper_bound'] - 
                                        region_forecast.loc[timestamp, 'lower_bound']) / 2
                    
                    # 使用权重来合成不确定性（权重越高，该省份对总体不确定性的贡献越大）
                    weighted_variance += (region_weight * region_uncertainty) ** 2
                    total_weight += region_weight
            
            # 归一化权重
            if total_weight > 0:
                weighted_variance = weighted_variance / (total_weight ** 2)
            
            # 计算总体不确定性
            total_uncertainty = np.sqrt(weighted_variance)
            
            # 设置融合结果
            fused_forecast.loc[timestamp, 'predicted'] = total_predicted
            fused_forecast.loc[timestamp, 'lower_bound'] = total_predicted - total_uncertainty
            fused_forecast.loc[timestamp, 'upper_bound'] = total_predicted + total_uncertainty
        
        print("区间预测融合完成（修正版本）:")
        print(f"  融合净负荷总和平均值: {fused_forecast['predicted'].mean():.1f} MW")
        print(f"  平均区间宽度: {(fused_forecast['upper_bound'] - fused_forecast['lower_bound']).mean():.1f} MW")
        print(f"  预测范围: [{fused_forecast['predicted'].min():.1f}, {fused_forecast['predicted'].max():.1f}] MW")
        
        # 验证：与各省份预测值之和的对比
        total_individual_sum = sum(forecast['predicted'].mean() for forecast in net_load_forecasts.values())
        print(f"  验证：各省份净负荷平均值之和: {total_individual_sum:.1f} MW")
        print(f"  验证：融合结果与求和的差异: {abs(fused_forecast['predicted'].mean() - total_individual_sum):.1f} MW")
        
        return fused_forecast
    
    def run_full_fusion(self, use_time_varying_weights: bool = True) -> Dict:
        """
        执行完整的多区域净负荷融合流程
        
        Args:
            use_time_varying_weights: 是否使用时变权重
            
        Returns:
            包含融合结果和相关信息的字典
        """
        if not self.regional_data:
            raise ValueError("请先加载区域预测数据")
        
        print("=" * 60)
        print("开始多区域净负荷预测融合")
        print("=" * 60)
        
        # 步骤1: 计算净负荷预测
        print("\n[步骤1] 计算各区域净负荷预测")
        net_load_forecasts = self.calculate_net_load_forecasts()
        
        # 步骤2: 计算融合权重
        print(f"\n[步骤2] 计算融合权重 (时变权重: {use_time_varying_weights})")
        weights = self.calculate_fusion_weights(net_load_forecasts, use_time_varying_weights)
        
        # 步骤3: 执行区间融合
        print("\n[步骤3] 执行区间预测融合")
        fused_net_load = self.execute_interval_fusion(net_load_forecasts, weights)
        
        # 步骤4: 汇总结果
        print("\n[步骤4] 汇总融合结果")
        
        # 计算各区域贡献度统计
        region_contributions = {}
        if isinstance(weights, pd.DataFrame):
            # 时变权重的平均贡献度
            for region in weights.columns:
                region_contributions[region] = {
                    'avg_weight': weights[region].mean(),
                    'weight_std': weights[region].std(),
                    'weight_range': [weights[region].min(), weights[region].max()]
                }
        else:
            # 静态权重
            for region, weight in weights.items():
                region_contributions[region] = {
                    'avg_weight': weight,
                    'weight_std': 0.0,
                    'weight_range': [weight, weight]
                }
        
        # 计算融合统计信息
        total_renewable_capacity = 0
        for region_data in self.regional_data.values():
            if region_data.pv_predictions is not None:
                total_renewable_capacity += region_data.pv_predictions.max()
            if region_data.wind_predictions is not None:
                total_renewable_capacity += region_data.wind_predictions.max()
        
        fusion_results = {
            'fused_net_load_forecast': fused_net_load,
            'regional_net_load_forecasts': net_load_forecasts,
            'fusion_weights': weights,
            'region_contributions': region_contributions,
            'fusion_statistics': {
                'total_regions': len(self.regional_data),
                'forecast_horizon': len(fused_net_load),
                'avg_net_load': fused_net_load['predicted'].mean(),
                'net_load_range': [fused_net_load['predicted'].min(), fused_net_load['predicted'].max()],
                'avg_interval_width': (fused_net_load['upper_bound'] - fused_net_load['lower_bound']).mean(),
                'total_renewable_capacity': total_renewable_capacity,
                'time_range': [fused_net_load.index[0], fused_net_load.index[-1]]
            },
            'method_info': {
                'fusion_method': 'PCA-based_weighted_interval_fusion',
                'uncertainty_propagation': 'stochastic_combination',
                'weight_type': 'time_varying' if use_time_varying_weights else 'static',
                'evaluation_metrics': ['ForecastReliability', 'ProvincialLoadImpact', 'ForecastingComplexity']
            }
        }
        
        self.fusion_results = fusion_results
        
        print("\n" + "=" * 60)
        print("多区域净负荷预测融合完成")
        print("=" * 60)
        print(f"融合了 {len(self.regional_data)} 个区域")
        print(f"预测时段: {len(fused_net_load)} 个时间点")
        print(f"净负荷范围: [{fused_net_load['predicted'].min():.1f}, {fused_net_load['predicted'].max():.1f}] MW")
        print(f"平均区间宽度: {(fused_net_load['upper_bound'] - fused_net_load['lower_bound']).mean():.1f} MW")
        
        print("\n区域权重贡献度:")
        for region, contrib in region_contributions.items():
            print(f"  {region}: {contrib['avg_weight']:.4f} (±{contrib['weight_std']:.4f})")
        
        return fusion_results
    
    def save_results(self, output_dir: str):
        """
        保存融合结果到文件
        
        Args:
            output_dir: 输出目录路径
        """
        if self.fusion_results is None:
            raise ValueError("请先执行融合计算")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存融合后的净负荷预测
        fused_forecast = self.fusion_results['fused_net_load_forecast']
        fused_forecast.to_csv(output_path / 'fused_net_load_forecast.csv')
        
        # 保存各区域净负荷预测
        for region, forecast in self.fusion_results['regional_net_load_forecasts'].items():
            forecast.to_csv(output_path / f'net_load_forecast_{region}.csv')
        
        # 保存权重信息
        weights = self.fusion_results['fusion_weights']
        if isinstance(weights, pd.DataFrame):
            weights.to_csv(output_path / 'fusion_weights_time_varying.csv')
        else:
            pd.Series(weights).to_csv(output_path / 'fusion_weights_static.csv')
        
        # 保存融合统计信息
        with open(output_path / 'fusion_summary.json', 'w', encoding='utf-8') as f:
            # 转换不可序列化的对象
            summary = self.fusion_results.copy()
            summary.pop('fused_net_load_forecast')
            summary.pop('regional_net_load_forecasts') 
            summary.pop('fusion_weights')
            
            # 转换时间戳
            if 'fusion_statistics' in summary and 'time_range' in summary['fusion_statistics']:
                time_range = summary['fusion_statistics']['time_range']
                summary['fusion_statistics']['time_range'] = [str(time_range[0]), str(time_range[1])]
            
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"融合结果已保存到: {output_path}")
        print("生成的文件:")
        print("  - fused_net_load_forecast.csv: 融合后的净负荷预测")
        print("  - net_load_forecast_{region}.csv: 各区域净负荷预测")
        print("  - fusion_weights_*.csv: 融合权重")
        print("  - fusion_summary.json: 融合统计摘要")


def example_usage():
    """使用示例"""
    print("多区域净负荷预测融合使用示例")
    print("-" * 40)
    
    # 1. 创建融合器
    fusion_system = MultiRegionalNetLoadFusion(
        smoothing_coefficient=0.7,
        smoothing_window=3,
        max_weight_factor=1.2,
        base_adjustment_coefficient=0.5
    )
    
    # 2. 加载区域预测数据
    json_files = {
        '上海': 'results/interval_forecast/load/上海/interval_load_上海_2024-05-26_20nineng_220219.json',
        '江苏': 'results/interval_forecast/load/江苏/interval_load_江苏_2024-05-26.json',
        '浙江': 'results/interval_forecast/load/浙江/interval_load_浙江_2024-05-26.json',
        '安徽': 'results/interval_forecast/load/安徽/interval_load_安徽_2024-05-26.json',
        '福建': 'results/interval_forecast/load/福建/interval_load_福建_2024-05-26.json'
    }
    
    try:
        fusion_system.load_from_json_files(json_files)
        
        # 3. 执行完整融合流程
        results = fusion_system.run_full_fusion(use_time_varying_weights=True)
        
        # 4. 保存结果
        fusion_system.save_results('results/multi_regional_fusion')
        
        print("\n融合完成！")
        return results
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保所有省份的预测结果文件都存在")
        return None
    except Exception as e:
        print(f"融合过程出错: {e}")
        return None


if __name__ == "__main__":
    # 运行示例
    results = example_usage() 