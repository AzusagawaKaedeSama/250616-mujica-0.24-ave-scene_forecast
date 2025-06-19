# This file will contain the PyTorch model definitions, refactored for the DDD architecture. 

import torch
import torch.nn as nn
import json
import os

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

class WeatherAwareConvTransformer:
    """
    一个封装了PyTorch模型加载和预测逻辑的类，充当了模型与其权重的"组装车间"。
    它本身不是一个nn.Module，而是一个管理器。
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
                raise FileNotFoundError(f"模型必需文件 '{name}' 未找到: {path}")
            
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

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        使用加载好的模型进行预测。
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output 