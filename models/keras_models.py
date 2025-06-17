# keras_models.py



import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, GRU # Keras imports for layers
from tensorflow.keras.models import Sequential, load_model # Keras imports for models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau # Callback for early stopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error # Evaluation metrics

from utils.scaler_manager import ScalerManager # Utility for managing data scaling
from data.dataset_builder import DatasetBuilder # Utility for building datasets
import pickle # For saving/loading scalers
import os # For directory operations
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import utils.plot_style as plot_style
from models.abstract_model import AbstractTimeSeriesModel

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")

from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout # <-- Moved Keras layers here
from tensorflow.keras.optimizers import RMSprop # <-- Optimizer import
from tensorflow.keras import mixed_precision # <-- Corrected import


import json
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
# 在代码开头设置策略
mixed_precision.set_global_policy('mixed_float16')
# 在代码开头
tf.config.optimizer.set_jit(False)
class KerasTimeSeriesModel:
    """时间序列模型的抽象基类"""
    def __init__(self, model_type, input_shape, learning_rate=0.01, dropout=0.2):
        self.model_type = model_type
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout = dropout  # 新增参数
        self.model = self.build_model()
        self.scaler = None
        # 初始化时自动编译
        self._compile()
    
    def _compile(self):
        """私有方法用于编译模型"""
        self.model.compile(
            optimizer=RMSprop(learning_rate=self.learning_rate),
            loss='mae',
            metrics=['mse'],
            run_eagerly=False
        )
        
    def build_model(self):
        """构建基础神经网络结构"""
        raise NotImplementedError("子类必须实现build_model方法")
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, 
          save_dir='models', callbacks=None):
        """训练模型"""

        # 定义学习率调度函数
        def lr_schedule(epoch):
            initial_lr = self.learning_rate
            if epoch < 50:  # 前50个epoch保持原学习率
                return initial_lr
            return initial_lr * tf.math.exp(0.1 * (50 - epoch))  # 指数衰减
    
        # 优化回调
        if callbacks is None:
            callbacks = []
            # 减少检查点保存频率
            model_checkpoint = ModelCheckpoint(
                f'{save_dir}/{self.model_type}_best.h5', 
                save_best_only=True,
                save_weights_only=True,  # 只保存权重可以加快速度
                monitor='val_loss'
            )
            # 早停策略 - 避免不必要的训练
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=7,
                min_delta=1e-4,
                restore_best_weights=True
            )
            lr_scheduler = LearningRateScheduler(lr_schedule)
            callbacks.extend([early_stop, model_checkpoint, lr_scheduler])
            
        # 显式编译模型
        self.model.compile(
            optimizer=RMSprop(learning_rate=self.learning_rate),
            loss='mae',
            metrics=['mse']
        )

        # 创建高效的数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # 训练模型
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return history
    


    def predict(self, X):
        """进行预测"""
        print(f"预测输入维度: {X.shape} (应匹配input_shape {self.input_shape})")
        return self.model.predict(X)

    def save(self, save_dir='models'):
        """保存模型和标准化器到指定目录"""
        os.makedirs(save_dir, exist_ok=True)  # 自动创建目录

        model_flag = shape_flag = scaler_flag = False

        # 使用SavedModel格式，更快且不易损坏
        model_path = os.path.join(save_dir, f'{self.model_type}_model')
        self.model.save(model_path, save_format='tf')

        # 保存输入形状（确保为列表格式）
        input_shape_path = os.path.join(save_dir, 'input_shape.json')
        with open(input_shape_path, 'w') as f:
            json.dump(list(self.input_shape), f)  # 转换为列表
            shape_flag = True
        
        # 保存标准化器
        if self.scaler is not None:
            scaler_path = os.path.join(save_dir, f'{self.model_type}_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            scaler_flag = True
        
        if model_flag and shape_flag :
            print(f"模型和输入形状已保存到目录: {save_dir}")
            if scaler_flag:
                print("标准化器已保存到目录")
        else:
            if model_flag:
                print("输入形状未保存")


    @classmethod
    def load(cls, save_dir='models'):
        model_type = cls.model_type
        
        # 定义可能的模型文件名模式
        possible_model_files = [
            # f'{model_type}_model.h5',  # 原始期望的文件名
            f'{model_type}_best.h5'   # ModelCheckpoint保存的文件名
            # f'model.h5',               # 通用备选名
            # f'best.h5'                 # 另一个通用备选名
        ]
        
        # 检查输入形状文件是否存在
        input_shape_path = os.path.join(save_dir, 'input_shape.json')
        if not os.path.exists(input_shape_path):
            raise FileNotFoundError(f"输入形状文件缺失: {input_shape_path}，请重新训练模型")
        
        # 查找任何一个可能的模型文件
        model_path = None
        for file_name in possible_model_files:
            path = os.path.join(save_dir, file_name)
            if os.path.exists(path):
                model_path = path
                print(f"找到模型文件: {path}")
                break
        
        if model_path is None:
            raise FileNotFoundError(f"在{save_dir}中未找到任何有效的模型文件，请重新训练模型")
        
        # 读取保存的输入形状
        with open(input_shape_path, 'r') as f:
            input_shape = json.load(f)
            
        # 创建模型实例
        instance = cls(input_shape=input_shape)
        
        # 首先尝试作为完整模型加载
        try:
            instance.model = load_model(model_path)
            print(f"作为完整模型成功加载: {model_path}")
        except Exception as e:
            print(f"无法作为完整模型加载({e})，尝试作为权重文件加载...")
            
            # 如果失败，尝试作为权重文件加载
            try:
                # 构建一个新模型
                instance.model = instance.build_model()
                # 编译模型
                instance._compile()
                # 加载权重
                instance.model.load_weights(model_path)
                print(f"作为权重文件成功加载: {model_path}")
            except Exception as load_weights_error:
                raise RuntimeError(f"加载模型失败! 完整模型错误: {e}, 权重加载错误: {load_weights_error}")
        

        
        # 尝试加载标准化器（如果存在）
        scaler = None
        possible_scaler_files = [
            f'{model_type}_scaler.pkl', 
            'scaler.pkl'
        ]
        
        for scaler_file in possible_scaler_files:
            scaler_path = os.path.join(save_dir, scaler_file)
            if os.path.exists(scaler_path):
                try:
                    scaler = joblib.load(scaler_path)
                    print(f"找到并加载标准化器: {scaler_path}")
                    break
                except Exception as e:
                    print(f"加载标准化器时出错: {e}")
        
        return instance


class KerasGRU(KerasTimeSeriesModel):
    model_type = 'gru'  # 新增类属性

    """基于GRU的时间序列模型"""
    def __init__(self, input_shape, learning_rate=0.01, dropout=0.2):
        super().__init__(model_type=self.model_type, input_shape=input_shape, learning_rate=learning_rate, dropout=dropout)
        self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(GRU(64, input_shape=self.input_shape, return_sequences=True, 
                    activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dropout(self.dropout))
        model.add(GRU(32, return_sequences=False,
                    activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        return model

class KerasLSTM(KerasTimeSeriesModel):
    model_type = 'lstm'  # 新增类属性

    """基于LSTM的时间序列模型"""
    def __init__(self, input_shape, learning_rate=0.01, dropout=0.2):
        super().__init__(model_type=self.model_type, input_shape=input_shape, learning_rate=learning_rate, dropout=dropout)
        self.build_model()
        
class KerasGRU(KerasTimeSeriesModel):
    model_type = 'gru'  # 新增类属性

    """基于GRU的时间序列模型"""
    def __init__(self, input_shape, learning_rate=0.01, dropout=0.2):
        super().__init__(model_type=self.model_type, input_shape=input_shape, learning_rate=learning_rate, dropout=dropout)
        self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(GRU(64, input_shape=self.input_shape, return_sequences=True, 
                    activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dropout(self.dropout))
        model.add(GRU(32, return_sequences=False,
                    activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        return model

class KerasLSTM(KerasTimeSeriesModel):
    model_type = 'lstm'  # 新增类属性

    """基于LSTM的时间序列模型"""
    def __init__(self, input_shape, learning_rate=0.01, dropout=0.2):
        super().__init__(model_type=self.model_type, input_shape=input_shape, learning_rate=learning_rate, dropout=dropout)
        self.build_model()
        
    def build_model(self):
        model = Sequential()
        # 第一层：LSTM输入层，返回序列
        model.add(LSTM(64, input_shape=self.input_shape, return_sequences=True,
                    activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dropout(self.dropout))
        
        # 第二层：LSTM中间层，不返回序列
        model.add(LSTM(32, return_sequences=False,
                    activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dropout(self.dropout))
        
        # 全连接层：先扩展特征，再压缩到输出
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(self.dropout/2))  # 轻微的dropout防止过拟合
        model.add(Dense(1))  # 最终输出层
        
        return model

class KerasModelFactory:
    """Keras模型工厂类"""
    @staticmethod
    def create_model(model_type, input_shape, **kwargs):
        """根据模型类型创建实例"""
        if model_type == 'gru':
            return KerasGRU(input_shape, **kwargs)
        elif model_type == 'lstm':
            return KerasLSTM(input_shape, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# 集成标准化处理
def train_with_standardization(models, X_train, y_train, X_val, y_val, scaler, 
                                epochs, batch_size, save_dir='models'):
    """训练流程集成标准化处理"""
    for model in models:
        # 应用标准化到输入数据
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 调整维度以适应模型输入
        X_train_scaled = X_train_scaled.reshape(*X_train_scaled.shape, 1)
        X_val_scaled = X_val_scaled.reshape(*X_val_scaled.shape, 1)
        
        # 训练模型
        history = model.train(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            epochs=epochs,
            batch_size=batch_size,
            save_dir=save_dir
        )
        
        # 保存训练好的模型
        model.save(save_dir=save_dir)
    
    return models

# 预测方法增强
def rolling_forecast_multimodal(models, initial_sequences, steps, scaler=None):
    """多模型滚动预测"""
    predictions_dict = {}
    for model in models:
        predictions = []
        current_seq = initial_sequences.copy()
        for _ in range(steps):
            pred = model.predict(current_seq.reshape(1, -1, 1))[0][0]
            if scaler is not None:
                pred = scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
            predictions.append(pred)
            current_seq = np.append(current_seq[1:], pred)
        predictions_dict[model.model_type] = predictions
    return predictions_dict