"""
相关变量不确定性传播的正确处理方法
基于协方差矩阵和蒙特卡洛模拟
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


class CorrelatedUncertaintyPropagation:
    """
    处理相关变量的不确定性传播
    """
    
    def __init__(self):
        self.correlation_matrices = {}
        self.historical_data = {}
    
    def estimate_correlations_from_data(self, 
                                      load_data: Dict[str, np.ndarray],
                                      pv_data: Dict[str, np.ndarray],
                                      wind_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        从历史数据估计相关性矩阵
        
        Args:
            load_data: 各省份负荷历史数据
            pv_data: 各省份光伏历史数据  
            wind_data: 各省份风电历史数据
            
        Returns:
            各省份的相关性矩阵字典
        """
        correlation_matrices = {}
        
        for province in load_data.keys():
            # 构建该省份的数据矩阵 [负荷, 光伏, 风电]
            data_matrix = np.column_stack([
                load_data[province],
                pv_data[province], 
                wind_data[province]
            ])
            
            # 计算相关系数矩阵
            corr_matrix = np.corrcoef(data_matrix.T)
            correlation_matrices[province] = corr_matrix
            
            print(f"{province} 相关性矩阵:")
            print(f"       负荷    光伏    风电")
            print(f"负荷  {corr_matrix[0,0]:.3f}  {corr_matrix[0,1]:.3f}  {corr_matrix[0,2]:.3f}")
            print(f"光伏  {corr_matrix[1,0]:.3f}  {corr_matrix[1,1]:.3f}  {corr_matrix[1,2]:.3f}")
            print(f"风电  {corr_matrix[2,0]:.3f}  {corr_matrix[2,1]:.3f}  {corr_matrix[2,2]:.3f}")
            print()
        
        return correlation_matrices
    
    def calculate_net_load_uncertainty_correct(self,
                                             load_mean: float, load_std: float,
                                             pv_mean: float, pv_std: float,
                                             wind_mean: float, wind_std: float,
                                             correlation_matrix: np.ndarray) -> Tuple[float, float]:
        """
        正确计算净负荷的均值和标准差（考虑相关性）
        
        净负荷 = 负荷 - 光伏 - 风电
        
        Args:
            load_mean, load_std: 负荷的均值和标准差
            pv_mean, pv_std: 光伏的均值和标准差
            wind_mean, wind_std: 风电的均值和标准差
            correlation_matrix: 3x3相关性矩阵 [负荷, 光伏, 风电]
            
        Returns:
            (净负荷均值, 净负荷标准差)
        """
        # 净负荷均值 = 负荷均值 - 光伏均值 - 风电均值
        net_load_mean = load_mean - pv_mean - wind_mean
        
        # 构建方差向量
        variances = np.array([load_std**2, pv_std**2, wind_std**2])
        
        # 构建权重向量 [1, -1, -1] (净负荷 = 负荷 - 光伏 - 风电)
        weights = np.array([1, -1, -1])
        
        # 计算协方差矩阵
        std_vector = np.array([load_std, pv_std, wind_std])
        cov_matrix = np.outer(std_vector, std_vector) * correlation_matrix
        
        # 净负荷方差 = w^T * Σ * w，其中w是权重向量，Σ是协方差矩阵
        net_load_variance = weights.T @ cov_matrix @ weights
        net_load_std = np.sqrt(max(0, net_load_variance))  # 确保非负
        
        return net_load_mean, net_load_std
    
    def monte_carlo_simulation(self,
                             means: np.ndarray,
                             stds: np.ndarray, 
                             correlation_matrix: np.ndarray,
                             n_samples: int = 10000) -> Tuple[float, float, np.ndarray]:
        """
        蒙特卡洛模拟验证相关变量的不确定性传播
        
        Args:
            means: 均值向量 [负荷均值, 光伏均值, 风电均值]
            stds: 标准差向量 [负荷标准差, 光伏标准差, 风电标准差]
            correlation_matrix: 相关性矩阵
            n_samples: 模拟样本数
            
        Returns:
            (净负荷均值, 净负荷标准差, 净负荷样本)
        """
        # 构建协方差矩阵
        cov_matrix = np.outer(stds, stds) * correlation_matrix
        
        # 生成多元正态分布样本
        samples = np.random.multivariate_normal(means, cov_matrix, n_samples)
        
        # 计算净负荷样本
        net_load_samples = samples[:, 0] - samples[:, 1] - samples[:, 2]
        
        # 统计结果
        net_load_mean_mc = np.mean(net_load_samples)
        net_load_std_mc = np.std(net_load_samples)
        
        return net_load_mean_mc, net_load_std_mc, net_load_samples
    
    def inter_provincial_correlation_analysis(self,
                                           provincial_net_loads: Dict[str, np.ndarray]) -> np.ndarray:
        """
        分析省际净负荷相关性
        
        Args:
            provincial_net_loads: 各省份净负荷历史数据
            
        Returns:
            省际相关性矩阵
        """
        provinces = list(provincial_net_loads.keys())
        n_provinces = len(provinces)
        
        # 构建省际数据矩阵
        data_matrix = np.column_stack([provincial_net_loads[p] for p in provinces])
        
        # 计算省际相关性矩阵
        inter_corr_matrix = np.corrcoef(data_matrix.T)
        
        # 可视化相关性矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(inter_corr_matrix, 
                   xticklabels=provinces, 
                   yticklabels=provinces,
                   annot=True, fmt='.3f', cmap='coolwarm', center=0)
        plt.title('省际净负荷相关性矩阵')
        plt.tight_layout()
        plt.show()
        
        return inter_corr_matrix
    
    def aggregate_with_correlation(self,
                                 provincial_means: Dict[str, float],
                                 provincial_stds: Dict[str, float],
                                 inter_corr_matrix: np.ndarray) -> Tuple[float, float]:
        """
        考虑省际相关性的正确聚合方法
        
        Args:
            provincial_means: 各省份净负荷均值
            provincial_stds: 各省份净负荷标准差
            inter_corr_matrix: 省际相关性矩阵
            
        Returns:
            (总净负荷均值, 总净负荷标准差)
        """
        provinces = list(provincial_means.keys())
        
        # 总均值 = 各省份均值之和
        total_mean = sum(provincial_means.values())
        
        # 构建标准差向量
        std_vector = np.array([provincial_stds[p] for p in provinces])
        
        # 构建协方差矩阵
        cov_matrix = np.outer(std_vector, std_vector) * inter_corr_matrix
        
        # 权重向量全为1（求和）
        weights = np.ones(len(provinces))
        
        # 总方差 = w^T * Σ * w
        total_variance = weights.T @ cov_matrix @ weights
        total_std = np.sqrt(total_variance)
        
        return total_mean, total_std


def demonstration_example():
    """
    演示正确的相关性处理方法
    """
    print("🎯 相关变量不确定性传播演示")
    print("=" * 60)
    
    # 模拟数据
    np.random.seed(42)
    
    # 假设参数
    load_mean, load_std = 20000, 2000
    pv_mean, pv_std = 500, 150
    wind_mean, wind_std = 300, 100
    
    # 假设相关性矩阵 [负荷, 光伏, 风电]
    correlation_matrix = np.array([
        [1.0,  0.3, -0.1],  # 负荷与光伏正相关，与风电负相关
        [0.3,  1.0,  0.0],  # 光伏与风电无相关
        [-0.1, 0.0,  1.0]   # 
    ])
    
    print("假设参数:")
    print(f"负荷: 均值={load_mean}MW, 标准差={load_std}MW")
    print(f"光伏: 均值={pv_mean}MW, 标准差={pv_std}MW") 
    print(f"风电: 均值={wind_mean}MW, 标准差={wind_std}MW")
    print("\n相关性矩阵:")
    print(correlation_matrix)
    
    # 创建分析器
    analyzer = CorrelatedUncertaintyPropagation()
    
    # 方法1：解析计算（考虑相关性）
    net_mean_correct, net_std_correct = analyzer.calculate_net_load_uncertainty_correct(
        load_mean, load_std, pv_mean, pv_std, wind_mean, wind_std, correlation_matrix
    )
    
    # 方法2：蒙特卡洛验证
    means = np.array([load_mean, pv_mean, wind_mean])
    stds = np.array([load_std, pv_std, wind_std])
    net_mean_mc, net_std_mc, samples = analyzer.monte_carlo_simulation(
        means, stds, correlation_matrix, n_samples=100000
    )
    
    # 方法3：错误的独立假设
    net_mean_wrong = load_mean - pv_mean - wind_mean  # 均值计算相同
    net_std_wrong = np.sqrt(load_std**2 + pv_std**2 + wind_std**2)  # 忽略相关性
    
    print("\n🔍 结果对比:")
    print("=" * 60)
    print(f"正确方法(解析): 均值={net_mean_correct:.1f}MW, 标准差={net_std_correct:.1f}MW")
    print(f"蒙特卡洛验证:   均值={net_mean_mc:.1f}MW, 标准差={net_std_mc:.1f}MW")
    print(f"错误方法(独立): 均值={net_mean_wrong:.1f}MW, 标准差={net_std_wrong:.1f}MW")
    
    print(f"\n标准差差异: {abs(net_std_correct - net_std_wrong):.1f}MW")
    print(f"相对误差: {abs(net_std_correct - net_std_wrong)/net_std_correct*100:.1f}%")
    
    # 区间对比
    confidence_level = 1.96  # 95%置信区间
    
    print(f"\n📊 95%置信区间对比:")
    correct_lower = net_mean_correct - confidence_level * net_std_correct
    correct_upper = net_mean_correct + confidence_level * net_std_correct
    
    wrong_lower = net_mean_wrong - confidence_level * net_std_wrong
    wrong_upper = net_mean_wrong + confidence_level * net_std_wrong
    
    print(f"正确方法: [{correct_lower:.1f}, {correct_upper:.1f}] MW")
    print(f"错误方法: [{wrong_lower:.1f}, {wrong_upper:.1f}] MW")
    print(f"区间宽度差异: {abs((correct_upper-correct_lower) - (wrong_upper-wrong_lower)):.1f}MW")


if __name__ == "__main__":
    demonstration_example() 