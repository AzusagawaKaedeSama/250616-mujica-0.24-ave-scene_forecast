"""
基于深度学习的直接总净负荷预测方法
避免相关性问题，端到端学习所有省份的净负荷总和
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class DirectNetLoadDataset(Dataset):
    """
    直接净负荷预测的数据集
    """
    
    def __init__(self, 
                 load_data: Dict[str, np.ndarray],  # 各省份负荷数据
                 pv_data: Dict[str, np.ndarray],    # 各省份光伏数据
                 wind_data: Dict[str, np.ndarray],  # 各省份风电数据
                 weather_data: Optional[np.ndarray] = None,  # 天气数据
                 sequence_length: int = 96,  # 输入序列长度(4天)
                 prediction_horizon: int = 96):  # 预测长度(1天)
        
        self.provinces = list(load_data.keys())
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 准备数据
        self.features, self.targets = self._prepare_data(
            load_data, pv_data, wind_data, weather_data
        )
        
        # 标准化
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.features = self.feature_scaler.fit_transform(self.features)
        self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
    
    def _prepare_data(self, load_data, pv_data, wind_data, weather_data):
        """
        准备训练数据
        """
        # 计算历史净负荷（目标变量）
        net_loads = {}
        for province in self.provinces:
            net_loads[province] = load_data[province] - pv_data[province] - wind_data[province]
        
        # 计算总净负荷（目标）
        total_net_load = sum(net_loads.values())
        
        # 构建特征矩阵：[负荷, 光伏, 风电] × 省份数
        feature_dim = len(self.provinces) * 3  # 每个省份3个特征
        if weather_data is not None:
            feature_dim += weather_data.shape[1]  # 添加天气特征
        
        n_samples = len(total_net_load) - self.sequence_length - self.prediction_horizon + 1
        
        features = np.zeros((n_samples, self.sequence_length, feature_dim))
        targets = np.zeros((n_samples, self.prediction_horizon))
        
        for i in range(n_samples):
            # 输入特征: 历史sequence_length个时间点的数据
            feature_idx = 0
            
            # 各省份负荷、光伏、风电数据
            for province in self.provinces:
                features[i, :, feature_idx] = load_data[province][i:i+self.sequence_length]
                features[i, :, feature_idx+1] = pv_data[province][i:i+self.sequence_length]
                features[i, :, feature_idx+2] = wind_data[province][i:i+self.sequence_length]
                feature_idx += 3
            
            # 天气数据
            if weather_data is not None:
                features[i, :, feature_idx:] = weather_data[i:i+self.sequence_length]
            
            # 目标: 未来prediction_horizon个时间点的总净负荷
            targets[i, :] = total_net_load[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon]
        
        # 重塑特征为2D (samples, sequence_length * feature_dim)
        features = features.reshape(n_samples, -1)
        
        return features, targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.targets[idx])


class DirectNetLoadPredictor(nn.Module):
    """
    直接预测总净负荷的深度学习模型
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 96,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class UncertaintyQuantification:
    """
    不确定性量化模块（基于深度集成）
    """
    
    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models = []
        self.trained = False
    
    def train_ensemble(self, 
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      input_dim: int,
                      epochs: int = 100):
        """
        训练模型集成
        """
        self.models = []
        
        for i in range(self.n_models):
            print(f"训练模型 {i+1}/{self.n_models}")
            
            # 创建模型
            model = DirectNetLoadPredictor(input_dim)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # 训练
            model = self._train_single_model(
                model, train_loader, val_loader, optimizer, criterion, epochs
            )
            
            self.models.append(model)
        
        self.trained = True
    
    def _train_single_model(self, model, train_loader, val_loader, optimizer, criterion, epochs):
        """
        训练单个模型
        """
        model.train()
        
        for epoch in range(epochs):
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            if epoch % 20 == 0:
                val_loss = self._validate(model, val_loader, criterion)
                print(f"  Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                      f"Val Loss = {val_loss:.4f}")
        
        return model
    
    def _validate(self, model, val_loader, criterion):
        """
        验证模型
        """
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        model.train()
        return val_loss / len(val_loader)
    
    def predict_with_uncertainty(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测并量化不确定性
        
        Returns:
            (均值预测, 标准差, 所有模型预测)
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X).numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)  # (n_models, n_samples, output_dim)
        
        # 计算统计量
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred, predictions


def create_demonstration_data(n_days: int = 30) -> Tuple[Dict, Dict, Dict, np.ndarray]:
    """
    创建演示数据
    """
    n_points = n_days * 96  # 15分钟数据点
    provinces = ['江苏', '上海', '浙江', '安徽', '福建']
    
    # 基础负荷模式（日周期性）
    time_points = np.arange(n_points)
    daily_pattern = 0.3 * np.sin(2 * np.pi * time_points / 96) + 0.7
    
    load_data = {}
    pv_data = {}
    wind_data = {}
    
    # 各省份数据（有相关性）
    base_loads = {'江苏': 90000, '上海': 20000, '浙江': 70000, '安徽': 50000, '福建': 40000}
    
    for province in provinces:
        # 负荷数据（有日周期 + 噪声）
        load_data[province] = (
            base_loads[province] * daily_pattern + 
            np.random.normal(0, base_loads[province] * 0.1, n_points)
        )
        
        # 光伏数据（日间高，夜间0，与负荷有相关性）
        pv_base = np.maximum(0, np.sin(2 * np.pi * (time_points % 96) / 96 - np.pi/2))
        pv_data[province] = (
            base_loads[province] * 0.02 * pv_base + 
            np.random.normal(0, base_loads[province] * 0.005, n_points)
        )
        pv_data[province] = np.maximum(0, pv_data[province])  # 非负
        
        # 风电数据（随机但有持续性）
        wind_noise = np.random.normal(0, 0.1, n_points)
        wind_smooth = np.convolve(wind_noise, np.ones(24)/24, mode='same')  # 平滑
        wind_data[province] = (
            base_loads[province] * 0.015 * (0.5 + wind_smooth) + 
            np.random.normal(0, base_loads[province] * 0.002, n_points)
        )
        wind_data[province] = np.maximum(0, wind_data[province])  # 非负
    
    # 简化天气数据
    weather_data = np.column_stack([
        25 + 10 * np.sin(2 * np.pi * time_points / (96 * 7)) + np.random.normal(0, 2, n_points),  # 温度
        60 + 20 * np.sin(2 * np.pi * time_points / 96) + np.random.normal(0, 5, n_points),  # 湿度
    ])
    
    return load_data, pv_data, wind_data, weather_data


def compare_methods():
    """
    对比传统方法与深度学习直接预测方法
    """
    print("🚀 深度学习直接预测 vs 传统加和方法对比")
    print("=" * 70)
    
    # 创建数据
    load_data, pv_data, wind_data, weather_data = create_demonstration_data(n_days=30)
    
    # 计算真实总净负荷
    provinces = list(load_data.keys())
    true_total_net_load = sum(
        load_data[p] - pv_data[p] - wind_data[p] for p in provinces
    )
    
    print(f"数据概况:")
    print(f"  时间长度: 30天 ({len(true_total_net_load)}个数据点)")
    print(f"  省份数量: {len(provinces)}")
    print(f"  总净负荷均值: {true_total_net_load.mean():.1f} MW")
    print(f"  总净负荷标准差: {true_total_net_load.std():.1f} MW")
    
    # 方法1：传统加和方法（忽略相关性）
    print("\n[方法1] 传统加和方法（忽略相关性）")
    provincial_stds = {}
    for province in provinces:
        net_load = load_data[province] - pv_data[province] - wind_data[province]
        provincial_stds[province] = net_load.std()
    
    # 错误的独立假设
    total_std_wrong = np.sqrt(sum(std**2 for std in provincial_stds.values()))
    print(f"  预测标准差(独立假设): {total_std_wrong:.1f} MW")
    print(f"  实际总体标准差: {true_total_net_load.std():.1f} MW")
    print(f"  误差: {abs(total_std_wrong - true_total_net_load.std()):.1f} MW")
    
    # 方法2：正确的相关性方法
    print("\n[方法2] 考虑相关性的分析方法")
    from correlation_analysis import CorrelatedUncertaintyPropagation
    
    analyzer = CorrelatedUncertaintyPropagation()
    
    # 构建省际相关性矩阵
    provincial_net_loads = {}
    for province in provinces:
        provincial_net_loads[province] = load_data[province] - pv_data[province] - wind_data[province]
    
    inter_corr_matrix = analyzer.inter_provincial_correlation_analysis(provincial_net_loads)
    
    # 计算正确的聚合不确定性
    provincial_means = {p: provincial_net_loads[p].mean() for p in provinces}
    provincial_stds = {p: provincial_net_loads[p].std() for p in provinces}
    
    total_mean_corr, total_std_corr = analyzer.aggregate_with_correlation(
        provincial_means, provincial_stds, inter_corr_matrix
    )
    
    print(f"  预测均值: {total_mean_corr:.1f} MW")
    print(f"  预测标准差(考虑相关性): {total_std_corr:.1f} MW")
    print(f"  实际总体标准差: {true_total_net_load.std():.1f} MW")
    print(f"  误差: {abs(total_std_corr - true_total_net_load.std()):.1f} MW")
    
    # 方法3：深度学习直接预测
    print("\n[方法3] 深度学习直接预测")
    print("  (注: 实际训练需要更长时间，这里展示框架)")
    
    # 创建数据集
    dataset = DirectNetLoadDataset(
        load_data, pv_data, wind_data, weather_data,
        sequence_length=96, prediction_horizon=96
    )
    
    print(f"  训练样本数: {len(dataset)}")
    print(f"  输入特征维度: {dataset.features.shape[1]}")
    
    # 优势分析
    print("\n📊 方法对比总结:")
    print("=" * 70)
    print("方法                    | 不确定性预测误差 | 主要优势")
    print("-" * 70)
    print(f"传统加和(独立假设)       | {abs(total_std_wrong - true_total_net_load.std()):.1f} MW          | 简单快速")
    print(f"相关性分析方法          | {abs(total_std_corr - true_total_net_load.std()):.1f} MW          | 理论正确")
    print("深度学习直接预测        | 待训练验证        | 端到端学习")
    
    print("\n💡 深度学习方法的独特优势:")
    print("  1. 自动学习所有变量间的复杂非线性关系")
    print("  2. 端到端优化，无需手工设计相关性矩阵")
    print("  3. 可以集成天气、时间等多种外部特征")
    print("  4. 通过模型集成自然量化预测不确定性")
    print("  5. 可以处理概念漂移和分布变化")


if __name__ == "__main__":
    compare_methods() 