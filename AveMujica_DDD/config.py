#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUJICA DDD系统配置文件
集中管理所有的配置参数
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    """数据库配置"""
    type: str = "file"  # file, sqlite, postgresql
    host: str = "localhost"
    port: int = 5432
    database: str = "mujica_ddd"
    username: str = ""
    password: str = ""
    
    # 文件系统配置
    base_directory: str = "."
    results_directory: str = "results"
    models_directory: str = "models"
    data_directory: str = "data"
    scenarios_directory: str = "scenarios"


@dataclass
class ModelConfig:
    """模型配置"""
    default_model_type: str = "WeatherAwareConvTransformer"
    
    # 深度学习模型参数
    seq_length: int = 96  # 输入序列长度（24小时 × 4个15分钟间隔）
    pred_length: int = 1  # 预测长度
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # 天气特征
    weather_features: List[str] = field(default_factory=lambda: [
        'temperature',    # 温度
        'humidity',       # 湿度
        'pressure',       # 气压
        'wind_speed',     # 风速
        'wind_direction', # 风向
        'precipitation',  # 降水量
        'solar_radiation' # 太阳辐射
    ])
    
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 10  # 早停patience
    
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class PredictionConfig:
    """预测配置"""
    default_historical_days: int = 7  # 默认历史数据天数
    default_confidence_level: float = 0.9  # 默认置信水平
    
    # 支持的省份
    supported_provinces: List[str] = field(default_factory=lambda: [
        "上海", "江苏", "浙江", "安徽", "福建"
    ])
    
    # 省份到文件名的映射
    province_mapping: Dict[str, str] = field(default_factory=lambda: {
        '上海': 'shanghai',
        '安徽': 'anhui', 
        '浙江': 'zhejiang',
        '江苏': 'jiangsu',
        '福建': 'fujian'
    })
    
    # 预测类型
    forecast_types: List[str] = field(default_factory=lambda: [
        "load", "pv", "wind"
    ])


@dataclass
class WeatherScenarioConfig:
    """天气场景配置"""
    # 场景配置文件路径
    scenarios_config_file: str = "weather_scenarios.json"
    
    # 默认场景配置
    default_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "MODERATE_NORMAL": {
            "description": "温和正常天气场景",
            "uncertainty_multiplier": 1.0,
            "temperature_range": [10, 25],
            "humidity_range": [40, 70]
        },
        "EXTREME_HOT_HUMID": {
            "description": "极端高温高湿天气场景", 
            "uncertainty_multiplier": 3.0,
            "temperature_range": [32, 45],
            "humidity_range": [80, 100]
        },
        "EXTREME_COLD": {
            "description": "极端寒冷天气场景",
            "uncertainty_multiplier": 2.5,
            "temperature_range": [-10, 0],
            "humidity_range": [30, 80]
        },
        "STORM_RAIN": {
            "description": "暴雨大风天气场景",
            "uncertainty_multiplier": 3.5,
            "wind_speed_min": 15,
            "precipitation_min": 20
        }
    })


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 5001
    debug: bool = True
    
    # CORS配置
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 300  # 缓存时间（秒）
    max_cache_size: int = 1000  # 最大缓存项数
    
    # 请求限制
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 300  # 5分钟超时


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 文件日志
    enable_file_logging: bool = True
    log_file: str = "logs/mujica_ddd.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # 控制台日志
    enable_console_logging: bool = True


@dataclass
class SystemConfig:
    """系统总配置"""
    # 各子系统配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    weather_scenario: WeatherScenarioConfig = field(default_factory=WeatherScenarioConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 系统信息
    system_name: str = "MUJICA DDD"
    version: str = "1.0.0"
    description: str = "天气感知负荷预测系统 - DDD架构版本"
    
    # 环境配置
    environment: str = "development"  # development, testing, production
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        self._ensure_directories()
        
        # 设置环境变量
        self._setup_environment()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.database.results_directory,
            self.database.models_directory,
            self.database.data_directory,
            self.database.scenarios_directory,
            Path(self.logging.log_file).parent,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_environment(self):
        """设置环境变量"""
        # 设置PyTorch线程数（可选）
        if not os.getenv('OMP_NUM_THREADS'):
            os.environ['OMP_NUM_THREADS'] = '4'
        
        # 设置CUDA可见设备（如果指定）
        if self.model.device.startswith('cuda:'):
            device_id = self.model.device.split(':')[1]
            os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    
    @classmethod
    def from_env(cls, env_prefix: str = "MUJICA_") -> 'SystemConfig':
        """从环境变量创建配置"""
        config = cls()
        
        # 从环境变量覆盖配置
        # 示例：MUJICA_API_PORT=8080
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_path = key[len(env_prefix):].lower().split('_')
                cls._set_nested_config(config, config_path, value)
        
        return config
    
    @staticmethod
    def _set_nested_config(config: 'SystemConfig', path: List[str], value: str):
        """设置嵌套配置值"""
        current = config
        
        # 遍历路径到最后一个属性
        for attr in path[:-1]:
            if hasattr(current, attr):
                current = getattr(current, attr)
            else:
                return  # 路径不存在，忽略
        
        # 设置最终值
        final_attr = path[-1]
        if hasattr(current, final_attr):
            # 尝试类型转换
            original_value = getattr(current, final_attr)
            if isinstance(original_value, bool):
                value = value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(original_value, int):
                value = int(value)
            elif isinstance(original_value, float):
                value = float(value)
            
            setattr(current, final_attr, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'database': self.database.__dict__,
            'model': self.model.__dict__,
            'prediction': self.prediction.__dict__,
            'weather_scenario': self.weather_scenario.__dict__,
            'api': self.api.__dict__,
            'logging': self.logging.__dict__,
            'system_name': self.system_name,
            'version': self.version,
            'description': self.description,
            'environment': self.environment,
        }


# 全局配置实例
config = SystemConfig()


def get_config() -> SystemConfig:
    """获取全局配置实例"""
    return config


def update_config(**kwargs) -> SystemConfig:
    """更新配置"""
    global config
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def load_config_from_file(config_file: str) -> SystemConfig:
    """从文件加载配置"""
    import json
    
    global config
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 更新配置
        for section, values in config_data.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, '__dict__'):
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
                else:
                    setattr(config, section, values)
    
    return config


def save_config_to_file(config_file: str) -> None:
    """保存配置到文件"""
    import json
    
    config_data = config.to_dict()
    
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)


# 预定义的配置环境
DEVELOPMENT_CONFIG = SystemConfig(
    environment="development",
    api=APIConfig(debug=True, port=5001),
    logging=LoggingConfig(level="DEBUG")
)

PRODUCTION_CONFIG = SystemConfig(
    environment="production",
    api=APIConfig(debug=False, port=8080, host="0.0.0.0"),
    logging=LoggingConfig(level="INFO", enable_console_logging=False)
)

TESTING_CONFIG = SystemConfig(
    environment="testing",
    database=DatabaseConfig(type="memory"),
    api=APIConfig(debug=True, port=5002),
    logging=LoggingConfig(level="DEBUG")
)


def get_config_for_environment(env: str) -> SystemConfig:
    """根据环境名获取配置"""
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "testing": TESTING_CONFIG,
    }
    
    return configs.get(env, DEVELOPMENT_CONFIG) 