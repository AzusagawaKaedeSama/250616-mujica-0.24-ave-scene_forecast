# multi_regional/model_based_forecast.py
import pandas as pd
import numpy as np
from data.data_loader import DataLoader
from data.dataset_builder import DatasetBuilder
from models.keras_models import KerasGRU, KerasLSTM
import os

class ModelBasedForecastGenerator:
    """使用训练好的模型生成多区域的预测数据"""
    
    def __init__(self, model_dirs=None):
        """
        初始化预测生成器
        
        Args:
            model_dirs: 模型保存目录的字典，例如 {'gru': 'models/gru', 'lstm': 'models/lstm'}
        """
        self.data_loader = DataLoader()
        self.dataset_builder = DatasetBuilder(self.data_loader)
        
        if model_dirs is None:
            self.model_dirs = {
                'gru': 'models/gru',
                'lstm': 'models/lstm'
            }
        else:
            self.model_dirs = model_dirs
            
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """加载预训练模型"""
        # 加载GRU模型
        gru_path = f"{self.model_dirs['gru']}/gru_model.h5"
        if os.path.exists(gru_path):
            print("加载已训练的GRU模型...")
            self.models['gru'] = KerasGRU.load(save_dir=self.model_dirs['gru'])
        else:
            print("GRU模型未找到")
        
        # 加载LSTM模型
        lstm_path = f"{self.model_dirs['lstm']}/lstm_model.h5"
        if os.path.exists(lstm_path):
            print("加载已训练的LSTM模型...")
            self.models['lstm'] = KerasLSTM.load(save_dir=self.model_dirs['lstm'])
        else:
            print("LSTM模型未找到")
    
    def train_models(self, train_start_date, train_end_date, epochs=50, batch_size=128):
        """
        训练模型
        
        Args:
            train_start_date: 训练数据开始日期
            train_end_date: 训练数据结束日期
            epochs: 训练周期数
            batch_size: 批量大小
        """
        # 训练数据集（Keras）
        train_data_keras = self.dataset_builder.build_for_date_range(
            train_start_date, train_end_date, 
            model_type='keras', 
            split=True
        )
        X_train_k, y_train_k, X_val_k, y_val_k = train_data_keras
        
        # 训练GRU模型
        print("训练GRU模型...")
        gru_model = KerasGRU(input_shape=X_train_k[0].shape)
        gru_model.train(
            X_train_k, y_train_k,
            X_val_k, y_val_k,
            epochs=epochs,
            batch_size=batch_size,
            save_dir=self.model_dirs['gru']
        )
        gru_model.save(save_dir=self.model_dirs['gru'])
        self.models['gru'] = gru_model
        
        # 训练LSTM模型
        print("训练LSTM模型...")
        lstm_model = KerasLSTM(input_shape=X_train_k[0].shape)
        lstm_model.train(
            X_train_k, y_train_k,
            X_val_k, y_val_k,
            epochs=epochs,
            batch_size=batch_size,
            save_dir=self.model_dirs['lstm']
        )
        lstm_model.save(save_dir=self.model_dirs['lstm'])
        self.models['lstm'] = lstm_model
        
        return self.models
    
    def generate_regional_forecasts(self, start_date, end_date, region_offsets=None, model_name='gru'):
        """
        为多个区域生成预测数据
        
        Args:
            start_date: 预测开始日期
            end_date: 预测结束日期
            region_offsets: 区域时间偏移的字典，如 {'Region1': 0, 'Region2': 1} 表示Region2使用的是1年前的数据
            model_name: 使用的模型名称，默认为'gru'
            
        Returns:
            实际数据和预测数据的字典
        """
        if region_offsets is None:
            region_offsets = {
                'Region1': 0,  # 当前年
                'Region2': 1,  # 1年前
                'Region3': 2,  # 2年前
                'Region4': 3,  # 3年前
                'Region5': 1,  # 1年前
                'Region6': 2   # 2年前
            }
        
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未加载")
        
        model = self.models[model_name]
        
        # 存储结果
        actual_data = {}
        forecast_data = {}
        
        # 预测起始日期 - 这个是最终结果的日期范围
        target_start = pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date)
        
        # 对每个区域生成预测数据
        for region, offset in region_offsets.items():
            # 计算偏移后的日期
            region_start = target_start.replace(year=target_start.year - offset)
            region_end = target_end.replace(year=target_end.year - offset)
            
            print(f"为 {region} 生成从 {region_start} 到 {region_end} 的预测数据")
            
            # 获取测试数据
            test_data = self.dataset_builder.build_for_date_range(
                region_start.strftime('%Y-%m-%d'), 
                region_end.strftime('%Y-%m-%d'),
                model_type='keras',
                split=False
            )
            X_test, y_test = test_data
            
            # 生成预测
            predictions = model.predict(X_test)
            
            # 保存实际数据
            actual_df = pd.DataFrame({'load': y_test.flatten()}, 
                                    index=pd.date_range(region_start, periods=len(y_test), freq='15min'))
            
            # 保存预测数据
            forecast_df = pd.DataFrame({'load': predictions.flatten()},
                                      index=pd.date_range(region_start, periods=len(predictions), freq='15min'))
            
            # 将索引调整为目标日期范围
            time_diff = target_start - region_start
            actual_df.index = actual_df.index + time_diff
            forecast_df.index = forecast_df.index + time_diff
            
            # 应用区域特性（可选）
            actual_df, forecast_df = self.apply_regional_characteristics(region, actual_df, forecast_df)
            
            # 存储结果
            actual_data[region] = actual_df
            forecast_data[region] = forecast_df
        
        return {
            "actual": actual_data,
            "forecast": forecast_data
        }
    
    def apply_regional_characteristics(self, region, actual_df, forecast_df):
        """
        应用区域特性（缩放比例、周末因子等）
        
        Args:
            region: 区域名称
            actual_df: 实际数据DataFrame
            forecast_df: 预测数据DataFrame
            
        Returns:
            修改后的实际数据和预测数据
        """
        # 定义区域特性
        region_characteristics = {
            "Region1": {"name": "Shanghai", "type": "service", "weekend_factor": 1.2, "scale": 0.85},
            "Region2": {"name": "Jiangsu", "type": "manufacturing", "weekend_factor": 0.95, "scale": 1.5},
            "Region3": {"name": "Zhejiang", "type": "mixed", "weekend_factor": 1.05, "scale": 1.1},
            "Region4": {"name": "Anhui", "type": "industrial", "weekend_factor": 0.9, "scale": 0.7},
            "Region5": {"name": "Fujian", "type": "coastal", "weekend_factor": 1.1, "scale": 0.8},
            "Region6": {"name": "Jiangxi", "type": "rural", "weekend_factor": 1.0, "scale": 0.6}
        }
        
        # 获取区域设置
        region_setting = region_characteristics.get(region, {"weekend_factor": 1.0, "scale": 1.0})
        
        # 应用周末因子
        actual_df['dayofweek'] = actual_df.index.dayofweek
        forecast_df['dayofweek'] = forecast_df.index.dayofweek
        
        weekend_mask_actual = (actual_df['dayofweek'] >= 5)  # 周六和周日
        weekend_mask_forecast = (forecast_df['dayofweek'] >= 5)  # 周六和周日
        
        actual_df.loc[weekend_mask_actual, 'load'] *= region_setting['weekend_factor']
        forecast_df.loc[weekend_mask_forecast, 'load'] *= region_setting['weekend_factor']
        
        # 应用比例因子
        actual_df['load'] *= region_setting['scale']
        forecast_df['load'] *= region_setting['scale']
        
        # 添加随机变化以表示区域特性
        np.random.seed(42 + list(region_characteristics.keys()).index(region))
        random_factor_actual = np.random.normal(1, 0.05, len(actual_df))
        random_factor_forecast = np.random.normal(1, 0.05, len(forecast_df))
        
        actual_df['load'] *= random_factor_actual
        forecast_df['load'] *= random_factor_forecast
        
        # 删除辅助列
        actual_df = actual_df.drop(columns=['dayofweek'])
        forecast_df = forecast_df.drop(columns=['dayofweek'])
        
        return actual_df, forecast_df