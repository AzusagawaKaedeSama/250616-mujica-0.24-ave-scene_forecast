import pandas as pd
import torch
import numpy as np
import os
import re

from AveMujica_DDD.application.ports.i_prediction_engine import IPredictionEngine
from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel
from AveMujica_DDD.domain.models.torch_models import WeatherAwareConvTransformer
from AveMujica_DDD.infrastructure.scalers.scaler_manager import ScalerManager
from AveMujica_DDD.infrastructure.feature_engineering.feature_engineering_service import DefaultFeatureEngineeringService

class PyTorchPredictionEngine(IPredictionEngine):
    """
    预测引擎接口的具体实现，使用PyTorch。
    它负责加载模型、调用特征工程、处理数据（包括标准化）、执行预测并返回真实值。
    """
    def __init__(self):
        self._feature_service = DefaultFeatureEngineeringService()

    def predict(self, model_aggregate: PredictionModel, input_data: pd.DataFrame) -> pd.Series:
        """
        加载并使用PyTorch模型执行一次点预测。
        """
        try:
            print(f"--- 使用PyTorch引擎和模型 '{model_aggregate.name}' v{model_aggregate.version} 进行预测 ---")
            
            # 1. 加载模型和 Scaler
            model_directory = os.path.dirname(model_aggregate.file_path)
            model_wrapper = WeatherAwareConvTransformer.load_from_directory(model_directory)
            scaler_directory = re.sub(r'models', r'models/scalers', model_directory, count=1)
            scaler_manager = ScalerManager(scaler_path=scaler_directory)

            # 2. 调用配置驱动的特征工程
            processed_features = self._feature_service.preprocess_for_prediction(
                data=input_data,
                target_column=model_aggregate.target_column,
                feature_columns=model_aggregate.feature_columns
            )
            
            # 3. 准备时间序列数据
            # 从配置中获取序列长度
            seq_length = model_wrapper.config.get('seq_length', 96)
            
            # 排除目标列，获取特征
            final_features = processed_features.drop(columns=[model_aggregate.target_column], errors='ignore')
            
            # 确保我们有足够的数据点来创建序列
            if len(final_features) < seq_length:
                raise ValueError(f"数据不足：需要至少 {seq_length} 个数据点，但只有 {len(final_features)} 个")
            
            # 取最后seq_length个数据点作为输入序列
            sequence_data = final_features.iloc[-seq_length:].values
            print(f"--- 序列数据形状: {sequence_data.shape} ---")
            
            # 4. 标准化输入数据
            # 将序列数据展平用于标准化
            sequence_flattened = sequence_data.reshape(1, -1)  # (1, seq_length * n_features)
            X_scaled_flat = scaler_manager.transform('X', sequence_flattened)
            
            # 重新reshape为序列形状
            X_scaled = X_scaled_flat.reshape(1, seq_length, -1)  # (1, seq_length, n_features)
            
            # 转换为PyTorch张量
            input_tensor = torch.from_numpy(X_scaled).float()
            print(f"--- 输入张量已创建并标准化，形状: {input_tensor.shape} ---")

            # 5. 执行推理
            output_tensor = model_wrapper.predict(input_tensor)
            print(f"--- 推理完成，输出张量形状: {output_tensor.shape} ---")

            # 6. 反标准化输出
            prediction_scaled = output_tensor.squeeze().cpu().numpy()
            
            # 确保prediction_scaled是2D的用于反标准化
            if prediction_scaled.ndim == 0:
                prediction_scaled = prediction_scaled.reshape(1, 1)
            elif prediction_scaled.ndim == 1:
                prediction_scaled = prediction_scaled.reshape(-1, 1)
                
            prediction_values = scaler_manager.inverse_transform('y', prediction_scaled).flatten()
            print(f"--- 预测结果已反标准化，预测值: {prediction_values} ---")
            
            # 7. 创建预测结果序列
            # 对于日前预测，我们通常预测未来24小时（96个点，每15分钟一个点）
            # 但是模型可能只输出一个点，我们需要根据实际情况调整
            
            if len(prediction_values) == 1:
                # 如果模型只预测一个点，我们为目标日期创建一个完整的序列
                # 这里我们简单地重复这个值（实际应用中可能需要更复杂的逻辑）
                target_date_start = input_data.index[-1] + pd.Timedelta(minutes=15)
                prediction_index = pd.date_range(
                    start=target_date_start,
                    periods=96,  # 一天96个点（15分钟间隔）
                    freq='15T'
                )
                # 简单重复预测值（实际应用中应该有更好的策略）
                prediction_series = pd.Series(
                    [prediction_values[0]] * 96,
                    index=prediction_index
                )
            else:
                # 如果模型预测多个点，直接使用
                target_date_start = input_data.index[-1] + pd.Timedelta(minutes=15)
                prediction_index = pd.date_range(
                    start=target_date_start,
                    periods=len(prediction_values),
                    freq='15T'
                )
                prediction_series = pd.Series(prediction_values, index=prediction_index)

            return prediction_series
            
        except Exception as e:
            print(f"!!! PyTorch预测引擎在执行时发生严重错误: {e}")
            raise 