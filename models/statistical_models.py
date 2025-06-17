# models/statistical_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class StatisticalTimeSeriesModel:
    """统计学时间序列模型的抽象基类"""
    def __init__(self, model_type, input_shape=None):
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.seasonal_periods = 96  # 默认为15分钟数据的日周期
        self.training_data = None  # 存储训练数据，用于滚动预测
    
    def build_model(self):
        """构建统计模型"""
        raise NotImplementedError("子类必须实现build_model方法")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=None, batch_size=None, save_dir='models', callbacks=None):
        """
        训练模型 - 为了保持与神经网络模型接口兼容，保留了额外参数
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            X_val: 验证数据特征 (对统计模型不使用)
            y_val: 验证数据标签 (对统计模型不使用)
            epochs, batch_size, callbacks: 为兼容神经网络接口保留
            save_dir: 模型保存目录
        """
        # 统计模型使用的是时间序列数据，将y_train转为时间序列
        if isinstance(y_train, np.ndarray):
            # 如果没有日期索引，创建一个人工索引
            ts_data = pd.Series(y_train.flatten())
        else:
            ts_data = y_train
        
        # 保存训练数据用于后续滚动预测
        self.training_data = ts_data
        
        self.model = self.build_model(ts_data)
        return self.model
    
    def predict(self, X):
        """
        进行预测
        
        Args:
            X: 输入数据，可以是形状为(样本数, 时间步长, 特征数)的神经网络样式输入
               或者是预测步数(用于统计模型)
        
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        # 如果X是三维数组(神经网络格式)，提取预测步数
        if isinstance(X, np.ndarray) and X.ndim > 1:
            steps = X.shape[0]
        else:
            steps = X
        
        # 执行预测，增加错误处理
        try:
            # 直接使用forecast方法
            forecast = self.model.forecast(steps)
            return forecast.values if isinstance(forecast, pd.Series) else forecast
        except ValueError as e:
            # 如果直接预测失败，尝试使用不同的方法
            print(f"直接预测失败: {e}")
            
            try:
                # 尝试使用predict方法
                if hasattr(self.model, 'predict'):
                    last_idx = getattr(self.model, 'nobs', 0)
                    forecast = self.model.predict(start=last_idx, end=last_idx + steps - 1)
                    return forecast.values if isinstance(forecast, pd.Series) else forecast
                
                # 对于Holt-Winters模型，可能需要其他方法
                elif hasattr(self.model, 'forecast'):
                    forecast = self.model.forecast(steps)
                    return forecast
                
                else:
                    raise ValueError("无法找到适当的预测方法")
            
            except Exception as e2:
                print(f"备选预测方法失败: {e2}")
                # 最后尝试进行重新训练后预测
                if self.training_data is not None:
                    print("尝试使用重新训练的模型进行预测")
                    self.model = self.build_model(self.training_data)
                    forecast = self.model.forecast(steps)
                    return forecast.values if isinstance(forecast, pd.Series) else forecast
                else:
                    raise ValueError("无法进行预测，模型需要重新训练")
    
    # 在StatisticalTimeSeriesModel类中替换rolling_predict方法

    def rolling_predict(self, test_data, window_size=96, step_size=96, update_model=True, max_attempts=3, timeout=300):
        """
        使用滚动窗口方法进行预测，增强版本
        
        Args:
            test_data: 测试数据（Pandas Series或DataFrame）
            window_size: 每个预测窗口大小（点数）
            step_size: 滚动步长（点数）
            update_model: 是否在每个窗口更新模型
            max_attempts: 每个窗口最大尝试次数
            timeout: 每个窗口预测超时时间(秒)
            
        Returns:
            预测结果（Pandas Series或numpy数组）
        """
        import signal
        from contextlib import contextmanager
        
        # 定义超时处理器
        @contextmanager
        def time_limit(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError("预测超时")
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
        
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        # 确保测试数据是Series对象
        if isinstance(test_data, np.ndarray):
            if test_data.ndim > 1:
                test_data = test_data.flatten()
            test_series = pd.Series(test_data)
        else:
            test_series = test_data
        
        # 创建结果容器
        results = pd.Series(index=test_series.index, dtype=float)
        
        # 获取训练数据的最后日期
        if self.training_data is not None and hasattr(self.training_data, 'index'):
            last_train_date = self.training_data.index[-1]
        else:
            # 如果没有训练数据日期信息，假设测试数据紧接在训练数据之后
            if isinstance(test_series.index[0], (pd.Timestamp, datetime)):
                freq = pd.infer_freq(test_series.index)
                last_train_date = test_series.index[0] - pd.Timedelta(seconds=1)
            else:
                last_train_date = None
        
        # 执行滚动预测
        total_steps = len(test_series)
        current_idx = 0
        
        print(f"开始滚动预测，总步数：{total_steps}，窗口大小：{window_size}，步长：{step_size}")
        
        while current_idx < total_steps:
            # 确定当前窗口的结束位置
            end_idx = min(current_idx + window_size, total_steps)
            
            # 当前窗口数据
            current_window = test_series.iloc[current_idx:end_idx]
            
            # 输出详细的进度信息
            if isinstance(current_window.index[0], (pd.Timestamp, datetime)):
                print(f"预测窗口 {current_idx // step_size + 1}/{(total_steps + step_size - 1) // step_size}: "
                    f"{current_window.index[0]} 到 {current_window.index[-1]} "
                    f"(点数: {len(current_window)})")
            else:
                print(f"预测窗口 {current_idx // step_size + 1}/{(total_steps + step_size - 1) // step_size}: "
                    f"{current_idx} 到 {end_idx-1} (点数: {len(current_window)})")
            
            # 如果需要更新模型，并且有训练数据
            if update_model and self.training_data is not None and current_idx > 0:
                try:
                    # 更新训练数据以包含之前的预测窗口
                    previous_window = test_series.iloc[:current_idx]
                    print(f"更新模型，增加 {len(previous_window)} 个新数据点")
                    
                    # 使用实际值而非预测值来更新模型
                    updated_training = pd.concat([self.training_data, previous_window])
                    self.model = self.build_model(updated_training)
                    self.training_data = updated_training
                    print(f"模型已使用新数据更新，训练数据长度: {len(updated_training)}")
                except Exception as e:
                    print(f"模型更新失败: {e}，使用原模型继续")
            
            # 尝试多次预测当前窗口
            window_pred = None
            success = False
            
            for attempt in range(max_attempts):
                try:
                    print(f"尝试预测窗口 {current_idx // step_size + 1} (尝试 {attempt+1}/{max_attempts})...")
                    
                    # 使用超时机制防止预测卡住
                    try:
                        with time_limit(timeout):
                            window_pred = self.predict(len(current_window))
                    except TimeoutError:
                        print(f"窗口 {current_idx // step_size + 1} 预测超时")
                        raise
                    
                    # 检查预测结果
                    if window_pred is None or np.isnan(window_pred).any():
                        raise ValueError("预测结果包含NaN或为None")
                    
                    # 确保预测结果长度匹配
                    if len(window_pred) > len(current_window):
                        window_pred = window_pred[:len(current_window)]
                        print(f"裁剪预测结果长度为 {len(current_window)}")
                    elif len(window_pred) < len(current_window):
                        # 填充不足部分
                        padding = np.zeros(len(current_window) - len(window_pred))
                        window_pred = np.concatenate([window_pred, padding])
                        print(f"填充预测结果长度为 {len(current_window)}")
                    
                    # 预测成功
                    success = True
                    print(f"窗口 {current_idx // step_size + 1} 预测成功")
                    break
                    
                except Exception as e:
                    print(f"窗口 {current_idx // step_size + 1} 尝试 {attempt+1} 预测失败: {e}")
                    
                    if attempt == max_attempts - 1:
                        # 最后一次尝试失败，使用备选方法
                        print(f"所有尝试均失败，使用备选填充方法")
                        
                        # 根据情况选择备选填充方法
                        if current_idx > 0:
                            # 使用前一个窗口的模式
                            last_pattern = results.iloc[max(0, current_idx-window_size):current_idx]
                            if len(last_pattern) > 0:
                                # 循环填充前一个窗口的模式
                                pattern_len = len(last_pattern)
                                window_pred = np.array([last_pattern.iloc[i % pattern_len] 
                                                    for i in range(len(current_window))])
                                print(f"使用前一个窗口的模式填充")
                            else:
                                # 使用训练数据的平均值
                                if self.training_data is not None:
                                    window_pred = np.full(len(current_window), self.training_data.mean())
                                    print(f"使用训练数据平均值填充: {self.training_data.mean()}")
                                else:
                                    # 最后手段：使用0填充
                                    window_pred = np.zeros(len(current_window))
                                    print("使用零值填充")
                        else:
                            # 第一个窗口，使用训练数据的日内模式
                            if self.training_data is not None and len(self.training_data) >= window_size:
                                # 使用最后一天的模式
                                last_day = self.training_data.iloc[-window_size:]
                                window_pred = last_day.values[:len(current_window)]
                                print("使用训练数据的最后一天模式填充")
                            else:
                                # 使用训练数据的均值
                                if self.training_data is not None:
                                    window_pred = np.full(len(current_window), self.training_data.mean())
                                    print(f"使用训练数据平均值填充: {self.training_data.mean()}")
                                else:
                                    # 最后手段：使用0填充
                                    window_pred = np.zeros(len(current_window))
                                    print("使用零值填充")
                    else:
                        # 尝试简化模型参数后重试
                        if hasattr(self, 'order') and attempt == 0:
                            print("尝试简化ARIMA参数后重试")
                            original_order = self.order
                            original_seasonal_order = getattr(self, 'seasonal_order', None)
                            
                            # 临时简化参数
                            self.order = (1, 1, 0)
                            if hasattr(self, 'seasonal_order'):
                                self.seasonal_order = (0, 1, 0, self.seasonal_periods)
                        elif attempt == 1:
                            # 如果是Holt-Winters模型
                            if hasattr(self, 'trend') and hasattr(self, 'seasonal'):
                                print("尝试简化Holt-Winters参数后重试")
                                # 备份原始参数
                                original_trend = self.trend
                                original_seasonal = self.seasonal
                                
                                # 临时简化参数
                                self.trend = None
                                self.seasonal = 'add'
            
            # 将结果存入总结果容器
            if window_pred is not None:
                results.iloc[current_idx:end_idx] = window_pred
                
                # 打印窗口预测基本统计信息
                pred_mean = np.mean(window_pred)
                pred_min = np.min(window_pred)
                pred_max = np.max(window_pred)
                print(f"窗口 {current_idx // step_size + 1} 预测统计: 均值={pred_mean:.2f}, 最小值={pred_min:.2f}, 最大值={pred_max:.2f}")
                
                # 计算当前窗口的性能指标
                window_actual = test_series.iloc[current_idx:end_idx]
                window_mae = np.mean(np.abs(window_actual - window_pred))
                window_mape = np.mean(np.abs((window_actual - window_pred) / window_actual)) * 100
                print(f"窗口 {current_idx // step_size + 1} 性能指标: MAE={window_mae:.4f}, MAPE={window_mape:.2f}%")
            
            # 恢复模型原始参数（如果曾经修改）
            if not success:
                if hasattr(self, 'order') and 'original_order' in locals():
                    self.order = original_order
                    if hasattr(self, 'seasonal_order') and 'original_seasonal_order' in locals() and original_seasonal_order:
                        self.seasonal_order = original_seasonal_order
                
                if hasattr(self, 'trend') and 'original_trend' in locals():
                    self.trend = original_trend
                    if hasattr(self, 'seasonal') and 'original_seasonal' in locals():
                        self.seasonal = original_seasonal
            
            # 更新索引位置，移动到下一个窗口
            print(f"完成窗口 {current_idx // step_size + 1}, 移动到下一个窗口\n")
            current_idx += step_size
        
        print(f"滚动预测完成，共预测 {len(results)} 个点")
        return results
    
    def update_model(self, new_data):
        """
        使用新数据更新模型
        
        Args:
            new_data: 新的训练数据
            
        Returns:
            更新后的模型
        """
        # 确保数据是Series对象
        if isinstance(new_data, np.ndarray):
            if new_data.ndim > 1:
                new_data = new_data.flatten()
            new_data = pd.Series(new_data)
        
        # 如果已有训练数据，合并旧数据和新数据
        if self.training_data is not None:
            combined_data = pd.concat([self.training_data, new_data])
        else:
            combined_data = new_data
        
        # 更新训练数据
        self.training_data = combined_data
        
        # 重新训练模型
        self.model = self.build_model(combined_data)
        return self.model
    
    def save(self, save_dir='models'):
        """保存模型到指定目录"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(save_dir, f'{self.model_type}_model.pkl')
        joblib.dump(self.model, model_path)
        
        # 保存训练数据
        if self.training_data is not None:
            data_path = os.path.join(save_dir, f'{self.model_type}_training_data.pkl')
            joblib.dump(self.training_data, data_path)
        
        # 保存配置信息
        config = {
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'seasonal_periods': self.seasonal_periods
        }
        config_path = os.path.join(save_dir, f'{self.model_type}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"{self.model_type}模型已保存到目录: {save_dir}")
    
    @classmethod
    def load(cls, save_dir='models'):
        """从指定目录加载模型"""
        model_type = cls.model_type
        
        # 检查文件是否存在
        model_path = os.path.join(save_dir, f'{model_type}_model.pkl')
        config_path = os.path.join(save_dir, f'{model_type}_config.json')
        data_path = os.path.join(save_dir, f'{model_type}_training_data.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"模型文件缺失，请重新训练模型")
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 创建模型实例
        model_instance = cls()
        model_instance.seasonal_periods = config.get('seasonal_periods', 96)
        
        # 加载模型
        model_instance.model = joblib.load(model_path)
        
        # 加载训练数据（如果存在）
        if os.path.exists(data_path):
            model_instance.training_data = joblib.load(data_path)
        
        return model_instance
    
    def decompose_series(self, series, plot=False, save_path=None):
        """
        对时间序列进行季节分解
        
        Args:
            series: 时间序列数据
            plot: 是否绘图
            save_path: 图片保存路径
        Returns:
            分解结果
        """
        # 确保数据是时间序列
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # 季节分解
        decompose_result = seasonal_decompose(
            series,
            model="additive",
            period=self.seasonal_periods
        )
        
        # 绘制分解结果
        if plot:
            plt.figure(figsize=(16, 12), dpi=300)
            
            # 趋势项
            plt.subplot(311)
            plt.plot(decompose_result.trend, color='#E74C3C')
            plt.title('趋势分量')
            
            # 季节项
            plt.subplot(312)
            plt.plot(decompose_result.seasonal[:self.seasonal_periods], color='#3498DB')
            plt.title('季节分量（单日展示）')
            
            # 残差项
            plt.subplot(313)
            plt.plot(decompose_result.resid, color='#2ECC71')
            plt.title('残差分量')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            else:
                plt.show()
        
        return decompose_result


class ARIMAModel(StatisticalTimeSeriesModel):
    """ARIMA时间序列模型"""
    model_type = 'arima'
    
    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 96), trend='c'):
        super().__init__(model_type=self.model_type)
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
    
    def build_model(self, ts_data):
        """构建ARIMA模型"""
        # 创建并拟合模型
        model = SARIMAX(
            ts_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend
        )
        
        # 拟合模型，增加收敛选项
        try:
            model_fit = model.fit(disp=False, maxiter=100)
        except Exception as e:
            print(f"ARIMA模型拟合失败：{e}，尝试使用更多迭代次数")
            try:
                # 使用更多迭代和低精度要求
                model_fit = model.fit(disp=False, maxiter=200, method='nm', low_memory=True)
            except Exception as e2:
                print(f"ARIMA模型二次拟合失败：{e2}，使用简化模型")
                # 再次减少复杂度尝试
                simple_model = SARIMAX(
                    ts_data,
                    order=(1, 1, 1),
                    seasonal_order=(0, 1, 0, self.seasonal_periods),
                    trend='c'
                )
                model_fit = simple_model.fit(disp=False)
        
        return model_fit
    
    def predict(self, steps):
        """
        为ARIMA模型重写predict方法，处理可能的错误
        
        Args:
            steps: 预测步数
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        try:
            # 方法1：使用forecast
            forecast = self.model.forecast(steps)
            return forecast
        except ValueError as e:
            print(f"forecast方法预测失败: {e}")
            
            try:
                # 方法2：使用get_prediction
                last_idx = getattr(self.model, 'nobs', 0)
                pred = self.model.get_prediction(start=last_idx, end=last_idx + steps - 1)
                return pred.predicted_mean
            except Exception as e2:
                print(f"get_prediction方法预测失败: {e2}")
                
                try:
                    # 方法3：使用predict
                    last_idx = getattr(self.model, 'nobs', 0)
                    forecast = self.model.predict(start=last_idx, end=last_idx + steps - 1)
                    return forecast
                except Exception as e3:
                    print(f"predict方法预测失败: {e3}，尝试重新拟合模型")
                    
                    # 最后手段：重新拟合模型
                    if self.training_data is not None:
                        # 重新拟合简化模型
                        simple_model = SARIMAX(
                            self.training_data,
                            order=(1, 1, 0),  # 简化的ARIMA参数
                            seasonal_order=(0, 1, 0, self.seasonal_periods),
                            trend='c'
                        )
                        self.model = simple_model.fit(disp=False)
                        return self.model.forecast(steps)
                    else:
                        raise ValueError("无法进行预测，所有预测方法均失败")
    
    def save(self, save_dir='models'):
        """保存ARIMA模型和参数"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(save_dir, f'{self.model_type}_model.pkl')
        joblib.dump(self.model, model_path)
        
        # 保存训练数据
        if self.training_data is not None:
            data_path = os.path.join(save_dir, f'{self.model_type}_training_data.pkl')
            joblib.dump(self.training_data, data_path)
        
        # 保存配置
        config = {
            'model_type': self.model_type,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'trend': self.trend,
            'seasonal_periods': self.seasonal_periods
        }
        config_path = os.path.join(save_dir, f'{self.model_type}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"ARIMA模型已保存到目录: {save_dir}")
    
    @classmethod
    def load(cls, save_dir='models'):
        """加载ARIMA模型"""
        model_type = cls.model_type
        config_path = os.path.join(save_dir, f'{model_type}_config.json')
        model_path = os.path.join(save_dir, f'{model_type}_model.pkl')
        data_path = os.path.join(save_dir, f'{model_type}_training_data.pkl')
        
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"ARIMA模型文件缺失，请重新训练模型")
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 创建模型实例
        model_instance = cls(
            order=tuple(config['order']),
            seasonal_order=tuple(config['seasonal_order']),
            trend=config['trend']
        )
        model_instance.seasonal_periods = config.get('seasonal_periods', 96)
        
        # 加载模型
        model_instance.model = joblib.load(model_path)
        
        # 加载训练数据（如果存在）
        if os.path.exists(data_path):
            model_instance.training_data = joblib.load(data_path)
        
        return model_instance


class HoltWintersModel(StatisticalTimeSeriesModel):
    """Holt-Winters指数平滑模型"""
    model_type = 'holtwinters'
    
    def __init__(self, trend='add', seasonal='add', seasonal_periods=96):
        super().__init__(model_type=self.model_type)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
    
    def build_model(self, ts_data):
        """构建Holt-Winters模型"""
        # 创建并拟合模型
        model = ExponentialSmoothing(
            ts_data,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method='estimated'
        )
        
        # 拟合模型，增加错误处理
        try:
            model_fit = model.fit(
                optimized=True,
                remove_bias=True,
                use_brute=True
            )
        except Exception as e:
            print(f"Holt-Winters模型拟合失败：{e}，尝试使用简化参数")
            try:
                # 使用简化参数
                model_fit = model.fit(
                    optimized=False,
                    remove_bias=False,
                    use_brute=False
                )
            except Exception as e2:
                print(f"Holt-Winters模型二次拟合失败：{e2}，使用默认初始值")
                # 使用默认初始值
                model = ExponentialSmoothing(
                    ts_data,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                    initialization_method='legacy-heuristic'
                )
                model_fit = model.fit()
        
        return model_fit
    
    def predict(self, steps):
        """
        为Holt-Winters模型重写predict方法，处理可能的错误
        
        Args:
            steps: 预测步数
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        try:
            # 标准预测方法
            forecast = self.model.forecast(steps)
            return forecast
        except Exception as e:
            print(f"标准预测方法失败: {e}")
            
            try:
                # 尝试使用predict方法（如果存在）
                if hasattr(self.model, 'predict'):
                    forecast = self.model.predict(start=self.model.nobs, end=self.model.nobs + steps - 1)
                    return forecast
                else:
                    raise AttributeError("模型没有predict方法")
            except Exception as e2:
                print(f"备选预测方法失败: {e2}")
                
                # 最后手段：重新拟合简化模型
                if self.training_data is not None:
                    print("重新拟合简化模型...")
                    simple_model = ExponentialSmoothing(
                        self.training_data,
                        trend=None,  # 简化参数，移除趋势项
                        seasonal=self.seasonal,
                        seasonal_periods=self.seasonal_periods,
                        initialization_method='legacy-heuristic'
                    )
                    self.model = simple_model.fit()
                    return self.model.forecast(steps)
                else:
                    raise ValueError("无法预测，所有方法均失败")
    
    def save(self, save_dir='models'):
        """保存Holt-Winters模型和参数"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(save_dir, f'{self.model_type}_model.pkl')
        joblib.dump(self.model, model_path)
        
        # 保存训练数据
        if self.training_data is not None:
            data_path = os.path.join(save_dir, f'{self.model_type}_training_data.pkl')
            joblib.dump(self.training_data, data_path)
        
        # 保存配置
        config = {
            'model_type': self.model_type,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods
        }
        config_path = os.path.join(save_dir, f'{self.model_type}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"Holt-Winters模型已保存到目录: {save_dir}")
    
    @classmethod
    def load(cls, save_dir='models'):
        """加载Holt-Winters模型"""
        model_type = cls.model_type
        config_path = os.path.join(save_dir, f'{model_type}_config.json')
        model_path = os.path.join(save_dir, f'{model_type}_model.pkl')
        data_path = os.path.join(save_dir, f'{model_type}_training_data.pkl')
        
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Holt-Winters模型文件缺失，请重新训练模型")
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 创建模型实例
        model_instance = cls(
            trend=config['trend'],
            seasonal=config['seasonal'],
            seasonal_periods=config['seasonal_periods']
        )
        
        # 加载模型
        model_instance.model = joblib.load(model_path)
        
        # 加载训练数据（如果存在）
        if os.path.exists(data_path):
            model_instance.training_data = joblib.load(data_path)
        
        return model_instance


class StatisticalModelFactory:
    """统计模型工厂类"""
    @staticmethod
    def create_model(model_type, **kwargs):
        """根据模型类型创建实例"""
        if model_type == 'arima':
            return ARIMAModel(**kwargs)
        elif model_type == 'holtwinters':
            return HoltWintersModel(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


def evaluate_statistical_model(model, test_data, forecast_steps=96, plot=True, save_path=None):
    """
    评估统计模型性能
    
    Args:
        model: 统计模型实例
        test_data: 测试数据
        forecast_steps: 预测步数
        plot: 是否绘图
        save_path: 图片保存路径
        
    Returns:
        评估指标
    """
    # 确保测试数据是Series
    if isinstance(test_data, np.ndarray):
        test_series = pd.Series(test_data.flatten())
    else:
        test_series = test_data
    
    # 进行预测
    forecast = model.predict(forecast_steps)
    
    # 确保预测结果和测试数据长度匹配
    min_len = min(len(forecast), len(test_series))
    forecast = forecast[:min_len]
    test_actual = test_series[:min_len]
    
    # 计算评估指标
    mape = np.mean(np.abs((test_actual - forecast) / test_actual)) * 100
    rmse = np.sqrt(np.mean((test_actual - forecast) ** 2))
    mae = np.mean(np.abs(test_actual - forecast))
    
    metrics = {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae
    }
    
    # 绘制预测结果
    if plot:
        plt.figure(figsize=(15, 6), dpi=300)
        
        # 绘制实际值
        plt.plot(test_series.index[:min_len], test_actual, 
                 label='实际值', color='#27AE60')
        
        # 绘制预测值
        plt.plot(test_series.index[:min_len], forecast, 
                 label='预测值', color='#E74C3C', linewidth=2)
        
        plt.title(f'{model.model_type.upper()}模型预测效果 (MAPE: {mape:.2f}%)')
        plt.xlabel('时间')
        plt.ylabel('负荷 (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
    
    return metrics


def evaluate_rolling_prediction(model, test_data, window_size=96, step_size=96, update_model=True, plot=True, save_path=None):
    """
    评估滚动预测性能
    
    Args:
        model: 统计模型实例
        test_data: 测试数据
        window_size: 滚动窗口大小
        step_size: 滚动步长
        update_model: 是否在每个窗口更新模型
        plot: 是否绘图
        save_path: 图片保存路径
        
    Returns:
        评估指标和预测结果
    """
    # 执行滚动预测
    forecast = model.rolling_predict(
        test_data, 
        window_size=window_size, 
        step_size=step_size, 
        update_model=update_model
    )
    
    # 确保测试数据是Series对象
    if isinstance(test_data, np.ndarray):
        test_series = pd.Series(test_data.flatten())
    else:
        test_series = test_data
    
    # 计算评估指标
    mape = np.mean(np.abs((test_series - forecast) / test_series)) * 100
    rmse = np.sqrt(np.mean((test_series - forecast) ** 2))
    mae = np.mean(np.abs(test_series - forecast))
    
    metrics = {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae
    }
    
    # 绘制预测结果
    if plot:
        plt.figure(figsize=(15, 6), dpi=300)
        
        # 绘制实际值
        plt.plot(test_series.index, test_series, 
                 label='实际值', color='#27AE60')
        
        # 绘制预测值
        plt.plot(forecast.index, forecast, 
                 label='预测值', color='#E74C3C', linewidth=2)
        
        plt.title(f'{model.model_type.upper()}模型滚动预测效果 (MAPE: {mape:.2f}%)')
        plt.xlabel('时间')
        plt.ylabel('负荷 (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
    
    return metrics, forecast