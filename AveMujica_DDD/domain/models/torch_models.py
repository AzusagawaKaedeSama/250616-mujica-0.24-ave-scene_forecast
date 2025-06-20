# This file will contain the PyTorch model definitions, refactored for the DDD architecture. 

import torch
import torch.nn as nn
import json
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod


class DomainModel(ABC):
    """
    领域模型抽象基类
    定义了所有模型必须实现的接口
    """
    
    @abstractmethod
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """执行预测"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass


class TemporalConvLSTM(nn.Module):
    """
    时空融合预测模型的核心网络结构。
    这是模型的 '大脑'，负责实际的计算。
    """
    def __init__(self, input_size, seq_len=96, pred_len=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 空间特征提取
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # 时间特征提取
        self.temporal_lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 预测头
        lstm_output_dim = 128 * 2
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim * (seq_len // 2), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq)
        x = self.spatial_conv(x)  # -> (batch, 64, seq//2)
        x = x.permute(0, 2, 1)  # -> (batch, seq//2, 64)
        lstm_out, _ = self.temporal_lstm(x)
        x = lstm_out.contiguous().view(x.size(0), -1)
        return self.fc(x)


class EnhancedConvTransformer(nn.Module):
    """
    增强的卷积Transformer模型
    结合了卷积和Transformer的优势
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        seq_length: int = 96,
        output_dim: int = 1,
        weather_features: int = 7,
        **kwargs
    ):
        super().__init__()
        
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'seq_length': seq_length,
            'output_dim': output_dim,
            'weather_features': weather_features,
            **kwargs
        }
        
        # 基础特征维度
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weather_features = weather_features
        
        # 特征嵌入层
        self.load_embedding = nn.Linear(input_dim - weather_features, hidden_dim)
        self.weather_embedding = nn.Linear(weather_features, hidden_dim)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(seq_length, hidden_dim)
        
        # 卷积层用于局部特征提取
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 融合注意力层
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 初始化参数
        self._initialize_weights()

    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """创建位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)

    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, input_dim)
            
        Returns:
            预测结果，形状为 (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 如果有天气特征，分离处理
        if self.weather_features > 0 and x.shape[-1] >= self.weather_features:
            load_features = x[:, :, :-self.weather_features]  # 负荷相关特征
            weather_features = x[:, :, -self.weather_features:]  # 天气特征
            
            # 特征嵌入
            load_embedded = self.load_embedding(load_features)
            weather_embedded = self.weather_embedding(weather_features)
            
            # 位置编码
            pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
            load_embedded = load_embedded + pos_enc
            weather_embedded = weather_embedded + pos_enc
            
            # 卷积特征提取
            load_conv = self.conv1d(load_embedded.transpose(1, 2)).transpose(1, 2)
            
            # Transformer编码
            load_encoded = self.transformer(load_conv)
            weather_encoded = self.transformer(weather_embedded)
            
            # 天气-负荷特征融合
            fused_features, _ = self.fusion_attention(
                query=load_encoded,
                key=weather_encoded,
                value=weather_encoded
            )
        else:
            # 没有天气特征时的简化处理
            embedded = self.load_embedding(x)
            pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
            embedded = embedded + pos_enc
            
            conv_features = self.conv1d(embedded.transpose(1, 2)).transpose(1, 2)
            fused_features = self.transformer(conv_features)
        
        # 取最后一个时间步的输出
        final_features = fused_features[:, -1, :]  # (B, H)
        
        # 输出预测
        output = self.output_projection(final_features)  # (B, output_dim)
        
        return output


class WeatherAwareConvTransformer(DomainModel):
    """
    一个封装了PyTorch模型加载和预测逻辑的类，充当了模型与其权重的"组装车间"。
    它本身不是一个nn.Module，而是一个管理器。
    现在实现了DomainModel接口。
    """
    def __init__(self, model_instance: nn.Module, config: dict, weather_config: dict, input_shape: tuple):
        self.model = model_instance
        self.config = config
        self.weather_config = weather_config
        self.input_shape = input_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @classmethod
    def load_from_directory(cls, directory_path: str):
        """
        从一个包含模型权重和多个配置文件的目录中，完整地加载并组装一个可用的模型实例。
        """
        print(f"--- [WeatherAwareConvTransformer] 正在从目录 '{directory_path}' 加载模型 ---")
        
        # 定义预期的文件路径
        # 根据旧脚本的逻辑，配置文件名可能是动态的
        config_path = os.path.join(directory_path, 'convtrans_weather_config.json')
        model_path = os.path.join(directory_path, 'best_model.pth')
        input_shape_path = os.path.join(directory_path, 'input_shape.json')
        weather_config_path = os.path.join(directory_path, 'weather_config.json')

        # 检查所有必需文件是否存在
        required_files = {'config': config_path, 'model': model_path, 'input_shape': input_shape_path}
        for name, path in required_files.items():
            if not os.path.exists(path):
                # 如果文件不存在，尝试使用默认配置创建新模型
                print(f"模型文件 '{name}' 未找到: {path}，将创建默认模型")
                return cls._create_default_model()
            
        try:
            # 1. 加载所有配置文件
            with open(config_path, 'r') as f: config = json.load(f)
            with open(input_shape_path, 'r') as f: input_shape = tuple(json.load(f))
            
            weather_config = {}
            if os.path.exists(weather_config_path):
                with open(weather_config_path, 'r') as f: weather_config = json.load(f)
            
            # 2. 根据配置创建模型实例
            input_size = input_shape[1] 
            seq_len = config.get('seq_length', 96)
            pred_len = 1 # 点预测

            # 优先使用增强的Transformer模型
            if config.get('model_type') == 'enhanced_transformer':
                model_instance = EnhancedConvTransformer(
                    input_dim=input_size,
                    seq_length=seq_len,
                    output_dim=pred_len,
                    **config
                )
            else:
                model_instance = TemporalConvLSTM(
                    input_size=input_size,
                    seq_len=seq_len,
                    pred_len=pred_len
                )
            
            # 3. 加载状态字典（权重）
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model_instance.load_state_dict(state_dict)
            
            print("--- 模型实例创建并加载权重成功 ---")

            # 4. 返回一个包含所有信息的完整封装器
            return cls(
                model_instance=model_instance, 
                config=config, 
                weather_config=weather_config,
                input_shape=input_shape
            )
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将创建默认模型")
            return cls._create_default_model()

    @classmethod
    def _create_default_model(cls):
        """创建默认模型实例"""
        print("创建默认模型配置...")
        
        # 默认配置
        config = {
            'seq_length': 96,
            'input_dim': 10,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.1,
            'model_type': 'enhanced_transformer'
        }
        
        input_shape = (1, 96, 10)  # (batch, seq, features)
        weather_config = {}
        
        # 创建增强的Transformer模型
        model_instance = EnhancedConvTransformer(**config)
        
        return cls(
            model_instance=model_instance,
            config=config,
            weather_config=weather_config,
            input_shape=input_shape
        )

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        使用加载好的模型进行预测。
        实现DomainModel接口。
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息。
        实现DomainModel接口。
        """
        return {
            'model_type': self.config.get('model_type', 'TemporalConvLSTM'),
            'config': self.config,
            'weather_config': self.weather_config,
            'input_shape': self.input_shape,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(self.device)
        }

    def save_to_directory(self, directory_path: str):
        """保存模型到目录"""
        os.makedirs(directory_path, exist_ok=True)
        
        # 保存模型权重
        model_path = os.path.join(directory_path, 'best_model.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # 保存配置文件
        config_path = os.path.join(directory_path, 'convtrans_weather_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        # 保存输入形状
        input_shape_path = os.path.join(directory_path, 'input_shape.json')
        with open(input_shape_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.input_shape), f)
        
        # 保存天气配置
        if self.weather_config:
            weather_config_path = os.path.join(directory_path, 'weather_config.json')
            with open(weather_config_path, 'w', encoding='utf-8') as f:
                json.dump(self.weather_config, f, indent=2, ensure_ascii=False)
        
        print(f"模型已保存到: {directory_path}")


def create_model_from_config(config: Dict[str, Any]) -> DomainModel:
    """
    工厂方法：根据配置创建模型
    
    Args:
        config: 模型配置
        
    Returns:
        创建的模型实例
    """
    model_type = config.get('model_type', 'WeatherAwareConvTransformer')
    
    if model_type == 'WeatherAwareConvTransformer':
        # 创建默认的WeatherAwareConvTransformer
        return WeatherAwareConvTransformer._create_default_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 为了向后兼容，保留旧的类名
PeakAwareConvTransformer = WeatherAwareConvTransformer 