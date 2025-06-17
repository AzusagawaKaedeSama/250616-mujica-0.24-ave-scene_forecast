#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气感知的ConvTransformer模型

基于PeakAwareConvTransformer架构，增加天气特征处理能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math

class WeatherAwareConvTransformer(nn.Module):
    """
    天气感知的ConvTransformer模型
    
    基于PeakAwareConvTransformer架构，增加天气特征处理能力：
    1. 保持原有的卷积-Transformer架构
    2. 增加天气特征编码器
    3. 融合天气信息到主预测流程
    4. 保持峰谷感知能力
    """
    
    def __init__(
        self,
        seq_length: int = 96,
        pred_length: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        weather_features: int = 6,  # 天气特征数量
        use_peak_loss: bool = True,
        peak_weight: float = 2.0,
        valley_weight: float = 1.5
    ):
        super(WeatherAwareConvTransformer, self).__init__()
        
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.d_model = d_model
        self.weather_features = weather_features
        self.use_peak_loss = use_peak_loss
        self.peak_weight = peak_weight
        self.valley_weight = valley_weight
        
        # 1. 负荷序列编码器（卷积层）
        self.load_conv_layers = nn.ModuleList([
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1)
        ])
        
        # 2. 天气特征编码器
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_features, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 3. 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout, seq_length)
        
        # 4. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # 5. 特征融合层
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 6. 峰谷感知层
        self.peak_aware_layer = nn.Sequential(
            nn.Linear(d_model + 2, d_model),  # +2 for peak/valley indicators
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 7. 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_length)
        )
        
        # 8. 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        load_seq: torch.Tensor,
        weather_features: torch.Tensor,
        peak_indicators: Optional[torch.Tensor] = None,
        valley_indicators: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            load_seq: 负荷序列 [batch_size, seq_length, 1]
            weather_features: 天气特征 [batch_size, seq_length, weather_features]
            peak_indicators: 峰时指示器 [batch_size, seq_length, 1]
            valley_indicators: 谷时指示器 [batch_size, seq_length, 1]
        
        Returns:
            预测结果 [batch_size, pred_length]
        """
        batch_size, seq_len, _ = load_seq.shape
        
        # 1. 负荷序列卷积编码
        load_conv = load_seq.transpose(1, 2)  # [batch_size, 1, seq_length]
        
        for conv_layer in self.load_conv_layers:
            load_conv = F.relu(conv_layer(load_conv))
        
        load_encoded = load_conv.transpose(1, 2)  # [batch_size, seq_length, d_model]
        
        # 2. 天气特征编码
        weather_encoded = self.weather_encoder(weather_features)  # [batch_size, seq_length, d_model]
        
        # 3. 位置编码
        load_encoded = self.pos_encoding(load_encoded)
        weather_encoded = self.pos_encoding(weather_encoded)
        
        # 4. Transformer编码
        load_transformed = self.transformer_encoder(load_encoded)
        weather_transformed = self.transformer_encoder(weather_encoded)
        
        # 5. 特征融合（使用注意力机制）
        fused_features, _ = self.fusion_layer(
            query=load_transformed,
            key=weather_transformed,
            value=weather_transformed
        )
        
        # 6. 峰谷感知处理
        if peak_indicators is not None and valley_indicators is not None:
            # 添加峰谷指示器
            peak_valley_info = torch.cat([peak_indicators, valley_indicators], dim=-1)
            fused_features = torch.cat([fused_features, peak_valley_info], dim=-1)
            fused_features = self.peak_aware_layer(fused_features)
        
        # 7. 全局池化（取最后一个时间步的特征）
        final_features = fused_features[:, -1, :]  # [batch_size, d_model]
        
        # 8. 预测
        predictions = self.prediction_head(final_features)  # [batch_size, pred_length]
        
        return predictions
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        peak_indicators: Optional[torch.Tensor] = None,
        valley_indicators: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算损失函数
        
        Args:
            predictions: 预测值 [batch_size, pred_length]
            targets: 真实值 [batch_size, pred_length]
            peak_indicators: 峰时指示器 [batch_size, pred_length]
            valley_indicators: 谷时指示器 [batch_size, pred_length]
        
        Returns:
            损失值
        """
        # 基础MSE损失
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        
        if self.use_peak_loss and peak_indicators is not None and valley_indicators is not None:
            # 峰谷加权损失
            weights = torch.ones_like(mse_loss)
            
            # 峰时权重
            peak_mask = peak_indicators.squeeze(-1) > 0.5
            weights[peak_mask] = self.peak_weight
            
            # 谷时权重
            valley_mask = valley_indicators.squeeze(-1) > 0.5
            weights[valley_mask] = self.valley_weight
            
            # 加权损失
            weighted_loss = mse_loss * weights
            return weighted_loss.mean()
        else:
            return mse_loss.mean()


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class WeatherAwareModelTrainer:
    """天气感知模型训练器"""
    
    def __init__(
        self,
        model: WeatherAwareConvTransformer,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # 获取批次数据
            load_seq = batch['load_seq'].to(self.device)
            weather_features = batch['weather_features'].to(self.device)
            targets = batch['targets'].to(self.device)
            peak_indicators = batch.get('peak_indicators', None)
            valley_indicators = batch.get('valley_indicators', None)
            
            if peak_indicators is not None:
                peak_indicators = peak_indicators.to(self.device)
            if valley_indicators is not None:
                valley_indicators = valley_indicators.to(self.device)
            
            # 前向传播
            predictions = self.model(
                load_seq=load_seq,
                weather_features=weather_features,
                peak_indicators=peak_indicators,
                valley_indicators=valley_indicators
            )
            
            # 计算损失
            loss = self.model.compute_loss(
                predictions=predictions,
                targets=targets,
                peak_indicators=peak_indicators,
                valley_indicators=valley_indicators
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取批次数据
                load_seq = batch['load_seq'].to(self.device)
                weather_features = batch['weather_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                peak_indicators = batch.get('peak_indicators', None)
                valley_indicators = batch.get('valley_indicators', None)
                
                if peak_indicators is not None:
                    peak_indicators = peak_indicators.to(self.device)
                if valley_indicators is not None:
                    valley_indicators = valley_indicators.to(self.device)
                
                # 前向传播
                predictions = self.model(
                    load_seq=load_seq,
                    weather_features=weather_features,
                    peak_indicators=peak_indicators,
                    valley_indicators=valley_indicators
                )
                
                # 计算损失
                loss = self.model.compute_loss(
                    predictions=predictions,
                    targets=targets,
                    peak_indicators=peak_indicators,
                    valley_indicators=valley_indicators
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 获取批次数据
                load_seq = batch['load_seq'].to(self.device)
                weather_features = batch['weather_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                peak_indicators = batch.get('peak_indicators', None)
                valley_indicators = batch.get('valley_indicators', None)
                
                if peak_indicators is not None:
                    peak_indicators = peak_indicators.to(self.device)
                if valley_indicators is not None:
                    valley_indicators = valley_indicators.to(self.device)
                
                # 前向传播
                predictions = self.model(
                    load_seq=load_seq,
                    weather_features=weather_features,
                    peak_indicators=peak_indicators,
                    valley_indicators=valley_indicators
                )
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        return np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)


def create_weather_aware_model(
    seq_length: int = 96,
    pred_length: int = 1,
    weather_features: int = 6,
    **kwargs
) -> WeatherAwareConvTransformer:
    """创建天气感知模型"""
    
    model_config = {
        'seq_length': seq_length,
        'pred_length': pred_length,
        'weather_features': weather_features,
        'd_model': kwargs.get('d_model', 512),
        'n_heads': kwargs.get('n_heads', 8),
        'n_layers': kwargs.get('n_layers', 3),
        'd_ff': kwargs.get('d_ff', 2048),
        'dropout': kwargs.get('dropout', 0.1),
        'activation': kwargs.get('activation', 'gelu'),
        'use_peak_loss': kwargs.get('use_peak_loss', True),
        'peak_weight': kwargs.get('peak_weight', 2.0),
        'valley_weight': kwargs.get('valley_weight', 1.5)
    }
    
    return WeatherAwareConvTransformer(**model_config) 