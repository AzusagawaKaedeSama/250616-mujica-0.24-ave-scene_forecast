import pandas as pd
import numpy as np
from datetime import datetime

from AveMujica_DDD.application.ports.i_prediction_engine import IPredictionEngine
from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel


class DummyPredictionEngine(IPredictionEngine):
    """
    一个假的IPredictionEngine实现，用于返回模拟的预测结果。
    它接收一个包含天气数据的DataFrame，并返回一个模拟的负荷预测序列。
    """
    def predict(
        self,
        model: PredictionModel,
        historical_and_future_data: pd.DataFrame
    ) -> pd.Series:
        """
        基于输入数据（特别是温度），生成一个模拟的负荷预测序列。
        """
        print(f"DummyPredictionEngine: Simulating prediction with model '{model.name}'")
        
        if 'temperature' not in historical_and_future_data.columns:
            raise ValueError("Input data for prediction must contain 'temperature' column.")
            
        # 模拟负荷与温度的正相关关系
        base_load = 25000
        temperature_effect = (historical_and_future_data['temperature'] - 15) * 500
        
        # 模拟周末负荷较低
        daily_multiplier = historical_and_future_data.index.dayofweek.to_series(index=historical_and_future_data.index).apply(
            lambda x: 0.85 if x >= 5 else 1.0
        )
        
        predicted_load = (base_load + temperature_effect) * daily_multiplier
        
        # 增加一些随机噪声
        noise = np.random.normal(0, 500, size=len(predicted_load))
        predicted_load += noise
        
        predicted_load.name = "predicted_value"
        
        return predicted_load 