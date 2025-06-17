"""
增强版数据集创建器
整合了原有DatasetBuilder功能和主函数中的数据准备逻辑
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import sys
sys.stdout.reconfigure(encoding='utf-8')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dataset_builder")

class DatasetBuilder:
    """增强版数据集构建器，支持训练、测试和滚动预测数据准备"""
    
    def __init__(self, data_loader=None, seq_length=96, pred_horizon=1, standardize=True):
        """
        初始化数据集构建器
        
        参数:
        data_loader: 数据加载器实例
        seq_length (int): 输入序列长度
        pred_horizon (int): 预测时间步长
        standardize (bool): 是否标准化数据
        """
        self.data_loader = data_loader
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.standardize = standardize
        self.scalers = {}
        # logger.info(f"DatasetBuilder初始化完成: 序列长度={seq_length}, 预测时长={pred_horizon}")
    
    def prepare_data_for_train(self, ts_data, test_ratio=0.2, model_type='torch', add_features=False, value_column='load'):
        """
        准备训练和验证数据
        
        参数:
        ts_data (DataFrame): 时间序列格式的数据
        test_ratio (float): 测试集比例
        model_type (str): 模型类型，'torch'或'keras'
        add_features (bool): 是否添加特征
        value_column (str): 值列的名称，如'load'、'pv'或'wind'
        
        返回:
        tuple: X_train, y_train, X_val, y_val
        """
        # 添加特征
        if add_features:
            # 确保ts_data是DataFrame
            if not isinstance(ts_data, pd.DataFrame):
                ts_data = pd.DataFrame({value_column: ts_data})
            
            # 使用build_dataset_from_df添加特征
            ts_data = self.build_dataset_from_df(ts_data, add_features=True, value_column=value_column)
        
        # 根据是否添加特征选择不同的处理路径
        if add_features and isinstance(ts_data, pd.DataFrame) and len(ts_data.columns) > 1:
            # 多特征路径
            if value_column in ts_data.columns:
                y_values = ts_data[value_column].values
                # 移除值列来构建特征矩阵
                X_values = ts_data.drop(value_column, axis=1).values
            else:
                y_values = ts_data.iloc[:, 0].values
                X_values = ts_data.iloc[:, 1:].values
                
            # 创建特征和标签序列
            X, y = [], []
            for i in range(len(y_values) - self.seq_length - self.pred_horizon + 1):
                # X序列包含所有特征
                X.append(X_values[i:i+self.seq_length])
                # y序列只包含目标变量
                y.append(y_values[i+self.seq_length:i+self.seq_length+self.pred_horizon])
        else:
            # 单特征路径（原始实现）
            if isinstance(ts_data, pd.DataFrame):
                if value_column in ts_data.columns:
                    data_values = ts_data[value_column].values
                else:
                    # 假设第一列是目标数据
                    data_values = ts_data.iloc[:, 0].values
            else:
                data_values = ts_data
                
            # 创建特征和标签
            X, y = [], []
            for i in range(len(data_values) - self.seq_length - self.pred_horizon + 1):
                X.append(data_values[i:i+self.seq_length])
                y.append(data_values[i+self.seq_length:i+self.seq_length+self.pred_horizon])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, self.pred_horizon)
        
        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 添加特征维度（如果是单特征）
        if not add_features or X_train.shape[-1] == 1:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        # 如果使用PyTorch格式且是单步预测，去掉最后一个维度
        if model_type == 'torch' and self.pred_horizon == 1:
            y_train = y_train.flatten()
            y_val = y_val.flatten()
        
        return X_train, y_train, X_val, y_val

    def prepare_data_for_test(self, ts_data, model_type='torch', add_features=False):
        """
        准备测试数据
        
        参数:
        ts_data (DataFrame): 时间序列格式的负荷数据
        model_type (str): 模型类型，'torch'或'keras'
        add_features (bool): 是否添加特征
        
        返回:
        tuple: X_test, y_test
        """
        # 添加特征
        if add_features:
            # 确保ts_data是DataFrame
            if not isinstance(ts_data, pd.DataFrame):
                ts_data = pd.DataFrame({'load': ts_data})
            
            # 使用build_dataset_from_df添加特征
            ts_data = self.build_dataset_from_df(ts_data, interval=1440/self.seq_length,add_features=True)
        
        # 根据是否添加特征选择不同的处理路径
        if add_features and isinstance(ts_data, pd.DataFrame) and len(ts_data.columns) > 1:
            # 多特征路径
            if 'load' in ts_data.columns:
                y_values = ts_data['load'].values
                # 移除load列来构建特征矩阵
                X_values = ts_data.drop('load', axis=1).values
            else:
                y_values = ts_data.iloc[:, 0].values
                X_values = ts_data.iloc[:, 1:].values
            
            # 创建特征和标签序列
            X, y = [], []
            for i in range(len(y_values) - self.seq_length - self.pred_horizon + 1):
                X.append(X_values[i:i+self.seq_length])
                y.append(y_values[i+self.seq_length:i+self.seq_length+self.pred_horizon])
        else:
            # 单特征路径（原始实现）
            if isinstance(ts_data, pd.DataFrame):
                if 'load' in ts_data.columns:
                    load_values = ts_data['load'].values
                else:
                    # 假设第一列是负荷数据
                    load_values = ts_data.iloc[:, 0].values
            else:
                load_values = ts_data
            
            # 创建特征和标签
            X, y = [], []
            for i in range(len(load_values) - self.seq_length - self.pred_horizon + 1):
                X.append(load_values[i:i+self.seq_length])
                y.append(load_values[i+self.seq_length:i+self.seq_length+self.pred_horizon])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, self.pred_horizon)
        
        # 添加特征维度（如果是单特征）
        if not add_features or X.shape[-1] == 1:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # 如果使用PyTorch格式且是单步预测，去掉最后一个维度
        if model_type == 'torch' and self.pred_horizon == 1:
            y = y.flatten()
        
        return X, y

    def prepare_data_for_rolling_forecast(self, ts_data, start_time, interval=15, model_type='torch', add_features=False, value_column='load'):
        """
        准备滚动预测的输入数据
        
        参数:
        ts_data (DataFrame): 时间序列格式的数据
        start_time (datetime): 开始预测的时间点
        interval (int): 数据间隔（分钟）
        model_type (str): 模型类型，'torch'或'keras'
        add_features (bool): 是否添加特征
        value_column (str): 值列的名称，如'load'、'pv'或'wind'
        
        返回:
        numpy.ndarray: 模型输入数据
        """
        # 找到起始时间之前的seq_length个点
        end_time = start_time - timedelta(minutes=interval)
        start_hist = end_time - timedelta(minutes=(self.seq_length-1)*interval)  # 计算历史开始时间
        
        # 添加特征
        if add_features:
            # 确保ts_data是DataFrame
            if not isinstance(ts_data, pd.DataFrame):
                ts_data = pd.DataFrame({value_column: ts_data})
            
            # 使用build_dataset_from_df添加特征
            all_data = self.build_dataset_from_df(ts_data, add_features=True, value_column=value_column)
            
            # 提取历史数据段
            historical_data = all_data.loc[start_hist:end_time]
            
            # 确保历史数据长度正确
            if len(historical_data) < self.seq_length:
                # print(f"警告：历史数据长度不足，只有{len(historical_data)}点，需要{self.seq_length}点")
                # 处理数据不足的情况，可以用前向或后向填充
                if len(historical_data) > 0:
                    # 如果有些数据点，则复制第一个点进行填充
                    first_row = historical_data.iloc[0:1].copy()
                    while len(historical_data) < self.seq_length:
                        # 在时间索引上调整
                        new_idx = historical_data.index[0] - (historical_data.index[1] - historical_data.index[0])
                        first_row.index = [new_idx]
                        historical_data = pd.concat([first_row, historical_data])
                else:
                    raise ValueError(f"没有足够的历史数据用于预测。需要至少1个数据点，但找到0个。")
            
            # 提取特征矩阵，排除值列
            if value_column in historical_data.columns:
                X_values = historical_data.drop(value_column, axis=1).values
            else:
                X_values = historical_data.iloc[:, 1:].values
            
            # 添加批次维度
            X = X_values.reshape(1, X_values.shape[0], X_values.shape[1])
        else:
            # 单特征路径
            if isinstance(ts_data, pd.DataFrame) and ts_data.index.dtype.kind == 'M':
                # 检查是否有日期时间索引
                historical_data = ts_data.loc[start_hist:end_time, value_column].values
            else:
                raise ValueError("输入数据必须有日期时间索引以便进行滚动预测")
            
            # 确保长度正确
            if len(historical_data) < self.seq_length:
                # 如果历史数据不足，用第一个值填充
                padding = np.full(self.seq_length - len(historical_data), 
                                historical_data[0] if len(historical_data) > 0 else 0)
                historical_data = np.concatenate([padding, historical_data])
            
            # 添加特征维度并创建batch维度
            X = historical_data.reshape(1, self.seq_length, 1)
        
        return X
    
    def fit_scalers(self, X_train, y_train):
        """
        拟合特征和目标的标准化器
        
        参数:
        X_train: 训练特征数据
        y_train: 训练目标数据
        
        返回:
        tuple: X标准化器, y标准化器
        """
        # 创建X标准化器
        X_scaler = StandardScaler()
        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_scaler.fit(X_reshaped)
        
        # 创建y标准化器
        y_scaler = StandardScaler()
        y_shaped = y_train.reshape(-1, 1) if len(y_train.shape) <= 1 else y_train
        y_scaler.fit(y_shaped)
        
        # 保存到实例
        self.scalers['X'] = X_scaler
        self.scalers['y'] = y_scaler
        
        return X_scaler, y_scaler

    def prepare_data_with_peak_awareness(self, ts_data, test_ratio=0.2, 
                                    peak_hours=(8, 20), valley_hours=(0, 6),
                                    peak_weight=2.0, valley_weight=1.5, 
                                    start_date=None, end_date=None, value_column='load'):
        """
        使用高峰和低谷感知特征准备训练和验证数据集

        参数:
        ts_data (DataFrame): 时间序列格式的数据
        test_ratio (float): 测试集比例
        peak_hours (tuple): 高峰时段的开始和结束小时
        valley_hours (tuple): 低谷时段的开始和结束小时
        peak_weight (float): 高峰时段的权重倍数
        valley_weight (float): 低谷时段的权重倍数
        start_date (str): 数据集的开始日期
        end_date (str): 数据集的结束日期
        value_column (str): 值列的名称，如'load'、'pv'或'wind'

        返回:
        tuple: X_train, y_train, X_val, y_val
        """
        # 调用 build_dataset_with_peak_awareness 增强数据
        enhanced_data = self.build_dataset_with_peak_awareness(
            df=ts_data,
            date_column='datetime',
            value_column=value_column,
            peak_hours=peak_hours,
            valley_hours=valley_hours,
            peak_weight=peak_weight,
            valley_weight=valley_weight,
            start_date=start_date,
            end_date=end_date
        )

        # 提取目标变量和特征
        y_values = enhanced_data[value_column].values
        X_values = enhanced_data.drop(columns=[value_column]).values

        # 创建特征和标签序列
        X, y = [], []
        for i in range(len(y_values) - self.seq_length - self.pred_horizon + 1):
            X.append(X_values[i:i + self.seq_length])
            y.append(y_values[i + self.seq_length:i + self.seq_length + self.pred_horizon])

        X = np.array(X)
        y = np.array(y).reshape(-1, self.pred_horizon)

        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 如果使用 PyTorch 格式且是单步预测，去掉最后一个维度
        if self.pred_horizon == 1:
            y_train = y_train.flatten()
            y_val = y_val.flatten()

        return X_train, y_train, X_val, y_val
    
    def build_dataset_from_df(self, df, date_column='datetime', value_column='load', 
                             start_date=None, end_date=None, interval = 15, add_features=False):
        """
        从DataFrame构建数据集
        
        参数:
        df (DataFrame): 包含时间和负荷数据的DataFrame
        date_column (str): 日期列名
        value_column (str): 值列名
        start_date: 开始日期
        end_date: 结束日期
        add_features (bool): 是否添加额外特征
        
        返回:
        DataFrame: 处理后的数据集
        """
        # 确保日期列是索引
        if date_column in df.columns:
            df = df.set_index(date_column)
        
        # 过滤日期范围
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        # 移除不需要的列，如PARTY_ID
        if 'PARTY_ID' in categorical_columns:
            df = df.drop(columns=['PARTY_ID'])
            categorical_columns.remove('PARTY_ID')

        # 如果不需要添加特征，直接返回
        if not add_features:
            return df
        
        # 添加时间特征
        result_df = df.copy()
        result_df['hour'] = result_df.index.hour
        result_df['day_of_week'] = result_df.index.dayofweek
        result_df['month'] = result_df.index.month
        result_df['is_weekend'] = (result_df.index.dayofweek >= 5).astype(int)
        
        # 添加周期性特征
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        
        # 添加滞后特征 - 使用动态列名
        result_df[f'{value_column}_lag_1'] = result_df[value_column].shift(1)
        result_df[f'{value_column}_lag_24'] = result_df[value_column].shift(int(24 * 60 / interval))  # 假设每小时1个点
        result_df[f'{value_column}_lag_168'] = result_df[value_column].shift(int(7 * 24 * 60 / interval))  # 一周前
        
        # 添加滑动窗口特征 - 使用动态列名
        result_df[f'{value_column}_rolling_mean_24'] = result_df[value_column].rolling(window=24).mean()
        result_df[f'{value_column}_rolling_std_24'] = result_df[value_column].rolling(window=24).std()
        
        # 填充缺失值
        result_df = result_df.fillna(method='bfill').fillna(method='ffill')
        
        return result_df
    
    def save_scalers(self, save_dir):
        """
        保存标准化器到指定目录
        
        参数:
        save_dir (str): 保存目录
        """
        import pickle
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        for name, scaler in self.scalers.items():
            with open(f"{save_dir}/{name}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
    
    def load_scalers(self, load_dir):
        """
        从指定目录加载标准化器
        
        参数:
        load_dir (str): 加载目录
        
        返回:
        bool: 是否成功加载
        """
        import pickle
        import os
        
        if not os.path.exists(load_dir):
            return False
        
        try:
            for name in ['X', 'y']:
                scaler_path = f"{load_dir}/{name}_scaler.pkl"
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers[name] = pickle.load(f)
                else:
                    return False
            return True
        except Exception as e:
            # print(f"加载标准化器时出错: {e}")
            return False
    
    def build_for_date_range(self, ts_data, start_date, end_date, model_type='torch', split=True):
        """
        为日期范围构建数据集
        
        参数:
        ts_data (DataFrame): 时间序列数据
        start_date: 开始日期
        end_date: 结束日期
        model_type (str): 模型类型，'torch'或'keras'
        split (bool): 是否划分训练和验证集
        
        返回:
        tuple: 根据split参数返回(X_train, y_train, X_val, y_val)或(X, y)
        """
        # 过滤日期范围
        filtered_data = ts_data.copy()
        if start_date:
            filtered_data = filtered_data[filtered_data.index >= pd.to_datetime(start_date)]
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= pd.to_datetime(end_date)]
        
        # 根据split参数选择合适的处理方法
        if split:
            return self.prepare_data_for_train(filtered_data, model_type=model_type)
        else:
            X, y = self.prepare_data_for_test(filtered_data, model_type=model_type)
            return X, y
    
    def enhance_peak_features_v0(self, df, peak_hours=(8, 20), workday_weight=2.0):
        """
        增强工作日高峰时段的特征
        
        参数:
        df (DataFrame): 包含时间索引的负荷数据
        peak_hours (tuple): 高峰时段的开始和结束小时，默认为(8, 20)
        workday_weight (float): 工作日高峰时段的权重倍数
        
        返回:
        DataFrame: 增强了高峰特征的数据集
        """
        enhanced_df = df.copy()
        
        # 确保有datetime索引
        if not isinstance(enhanced_df.index, pd.DatetimeIndex):
            raise ValueError("输入DataFrame必须有datetime索引")
        
        # 添加工作日标志（0=周末，1=工作日）
        enhanced_df['is_workday'] = (enhanced_df.index.dayofweek < 5).astype(int)
        
        # 添加高峰时段标志
        peak_start, peak_end = peak_hours
        enhanced_df['is_peak_hour'] = ((enhanced_df.index.hour >= peak_start) & 
                                    (enhanced_df.index.hour <= peak_end)).astype(int)
        
        # 创建工作日高峰特征
        enhanced_df['workday_peak'] = enhanced_df['is_workday'] * enhanced_df['is_peak_hour']
        
        # 添加高峰负荷的历史特征（前一天同时段、前一周同时段）
        if 'load' in enhanced_df.columns:
            # 前一天同时段负荷
            enhanced_df['load_prev_day_same_hour'] = enhanced_df['load'].shift(24 * 4)  # 假设15分钟间隔，24小时=96点
            
            # 前一周同时段负荷
            enhanced_df['load_prev_week_same_hour'] = enhanced_df['load'].shift(7 * 24 * 4)  # 7天前同一时刻
            
            # 计算高峰时段的移动平均和标准差
            enhanced_df['peak_load_ma'] = enhanced_df.loc[enhanced_df['is_peak_hour'] == 1, 'load'].rolling(window=4).mean()
            enhanced_df['peak_load_std'] = enhanced_df.loc[enhanced_df['is_peak_hour'] == 1, 'load'].rolling(window=4).std()
        
        # 对工作日高峰时段的特征赋予更高权重
        for col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
            if col in enhanced_df.columns:
                # 为工作日高峰时段特征增加权重
                enhanced_df[f'{col}_peak'] = enhanced_df[col] * (1 + (workday_weight - 1) * enhanced_df['workday_peak'])
        
        # 填充缺失值
        enhanced_df = enhanced_df.bfill().ffill()
        
        return enhanced_df
    
    def enhance_peak_features(self, df, interval=15, peak_hours=(8, 20), valley_hours=(0, 6), 
                         peak_weight=2.0, valley_weight=1.5, value_column='load'):
        """
        增强工作日高峰和低谷时段的特征
        
        参数:
        df (DataFrame): 包含时间索引的负荷数据
        peak_hours (tuple): 高峰时段的开始和结束小时，默认为(8, 20)
        valley_hours (tuple): 低谷时段的开始和结束小时，默认为(0, 6)
        peak_weight (float): 高峰时段的权重倍数
        valley_weight (float): 低谷时段的权重倍数
        value_column (str): 值列的名称，如'load'、'pv'或'wind'
        
        返回:
        DataFrame: 增强了高峰和低谷特征的数据集
        """
        enhanced_df = df.copy()
        
        # 确保有datetime索引
        if not isinstance(enhanced_df.index, pd.DatetimeIndex):
            raise ValueError("输入DataFrame必须有datetime索引")
        
        # 添加工作日标志（0=周末，1=工作日）
        enhanced_df['is_workday'] = (enhanced_df.index.dayofweek < 5).astype(int)
        
        # 添加高峰时段标志
        peak_start, peak_end = peak_hours
        enhanced_df['is_peak'] = ((enhanced_df.index.hour >= peak_start) & 
                                (enhanced_df.index.hour <= peak_end)).astype(int)
        
        # 添加低谷时段标志
        valley_start, valley_end = valley_hours
        enhanced_df['is_valley'] = ((enhanced_df.index.hour >= valley_start) & 
                                    (enhanced_df.index.hour <= valley_end)).astype(int)
        
        # 创建工作日高峰和低谷特征
        enhanced_df['workday_peak'] = enhanced_df['is_workday'] * enhanced_df['is_peak']
        enhanced_df['valley'] = enhanced_df['is_valley']
        
        # 添加高峰和低谷负荷的历史特征
        if value_column in enhanced_df.columns:
            # 前一天同时段负荷
            shift_prev_day = int(1440 / interval)
            enhanced_df[f'{value_column}_prev_day_same_hour'] = enhanced_df[value_column].shift(shift_prev_day)  # 假设15分钟间隔，24小时=96点
            
            # 前一周同时段负荷
            shift_prev_week = int(10080 /interval)
            enhanced_df[f'{value_column}_prev_week_same_hour'] = enhanced_df[value_column].shift(shift_prev_week)  # 7天前同一时刻
            
            # 计算高峰时段的移动平均和标准差
            peak_mask = enhanced_df['is_peak'] == 1
            if peak_mask.any():
                enhanced_df.loc[peak_mask, f'peak_{value_column}_ma'] = enhanced_df.loc[peak_mask, value_column].rolling(window=4).mean()
                enhanced_df.loc[peak_mask, f'peak_{value_column}_std'] = enhanced_df.loc[peak_mask, value_column].rolling(window=4).std()
            
            # 计算低谷时段的移动平均和标准差
            valley_mask = enhanced_df['is_valley'] == 1
            if valley_mask.any():
                enhanced_df.loc[valley_mask, f'valley_{value_column}_ma'] = enhanced_df.loc[valley_mask, value_column].rolling(window=4).mean()
                enhanced_df.loc[valley_mask, f'valley_{value_column}_std'] = enhanced_df.loc[valley_mask, value_column].rolling(window=4).std()
        
        # 添加时间特征
        enhanced_df['hour'] = enhanced_df.index.hour
        enhanced_df['day_of_week'] = enhanced_df.index.dayofweek
        enhanced_df['month'] = enhanced_df.index.month
        enhanced_df['is_weekend'] = (enhanced_df.index.dayofweek >= 5).astype(int)
        
        # 添加周期性特征
        enhanced_df['hour_sin'] = np.sin(2 * np.pi * enhanced_df['hour'] / 24)
        enhanced_df['hour_cos'] = np.cos(2 * np.pi * enhanced_df['hour'] / 24)
        enhanced_df['day_sin'] = np.sin(2 * np.pi * enhanced_df['day_of_week'] / 7)
        enhanced_df['day_cos'] = np.cos(2 * np.pi * enhanced_df['day_of_week'] / 7)
        
        # 对高峰和低谷时段的特征赋予权重
        for col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
            if col in enhanced_df.columns:
                # 为工作日高峰时段特征增加权重
                enhanced_df[f'{col}_peak'] = enhanced_df[col] * (1 + (peak_weight - 1) * enhanced_df['workday_peak'])
                # 为工作日低谷时段特征增加权重（较弱）
                enhanced_df[f'{col}_valley'] = enhanced_df[col] * (1 + (valley_weight - 1) * enhanced_df['valley'])
        
        # 添加高峰时段特有的特征
        # 1. 距离高峰开始/结束的时间
        enhanced_df['hours_to_peak_start'] = np.minimum(
            (enhanced_df['hour'] - peak_start) % 24,
            (peak_start - enhanced_df['hour']) % 24
        )
        enhanced_df['hours_to_peak_end'] = np.minimum(
            (enhanced_df['hour'] - peak_end) % 24,
            (peak_end - enhanced_df['hour']) % 24
        )
        
        # 2. 添加距离低谷开始/结束的时间
        enhanced_df['hours_to_valley_start'] = np.minimum(
            (enhanced_df['hour'] - valley_start) % 24,
            (valley_start - enhanced_df['hour']) % 24
        )
        enhanced_df['hours_to_valley_end'] = np.minimum(
            (enhanced_df['hour'] - valley_end) % 24,
            (valley_end - enhanced_df['hour']) % 24
        )
        
        # 3. 高峰和低谷负荷趋势
        if value_column in enhanced_df.columns:
            # 计算一阶差分（负荷变化率）
            enhanced_df[f'{value_column}_diff'] = enhanced_df[value_column].diff()
            
            # 为高峰时段创建特殊的负荷变化率特征
            enhanced_df[f'peak_{value_column}_diff'] = enhanced_df[f'{value_column}_diff'] * enhanced_df['is_peak']
            
            # 为低谷时段创建特殊的负荷变化率特征
            enhanced_df[f'valley_{value_column}_diff'] = enhanced_df[f'{value_column}_diff'] * enhanced_df['is_valley']
            
            # 计算高峰时段内的累积负荷变化
            enhanced_df[f'peak_cumulative_diff'] = enhanced_df[f'peak_{value_column}_diff'].cumsum()
            enhanced_df.loc[~peak_mask, f'peak_cumulative_diff'] = 0
            
            # 计算低谷时段内的累积负荷变化
            enhanced_df[f'valley_cumulative_diff'] = enhanced_df[f'valley_{value_column}_diff'].cumsum()
            enhanced_df.loc[~valley_mask, f'valley_cumulative_diff'] = 0
        
        # 4. 低谷特定时段特征
        enhanced_df['is_deep_valley'] = ((enhanced_df['hour'] >= 2) & 
                                        (enhanced_df['hour'] <= 6)).astype(int)  # 凌晨2-6点通常是最低负荷
        
        # 填充缺失值
        enhanced_df = enhanced_df.bfill().ffill()
        
        return enhanced_df

    def build_dataset_with_peak_awareness(self, df, date_column='datetime', value_column='load', interval=15,
                                    peak_hours=(9, 20), valley_hours=(0, 6),
                                    peak_weight=2.0, valley_weight=1.5,
                                    start_date=None, end_date=None):
        """
        构建带有峰值感知特征的数据集
        :param df: 输入数据框
        :param date_column: 日期列名
        :param value_column: 值列名
        :param interval: 数据间隔（分钟）
        :param peak_hours: 峰值小时范围（开始时间，结束时间）
        :param valley_hours: 谷值小时范围（开始时间，结束时间）
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 增强特征数据集
        """
        # 复制数据框以避免修改原始数据
        df = df.copy()

        # 处理PARTY_ID列，将其转换为数值特征而不是删除
        if 'PARTY_ID' in df.columns:
            # 将PARTY_ID转换为数值类型
            try:
                # 方法1：如果PARTY_ID已经是数值，直接使用
                if pd.api.types.is_numeric_dtype(df['PARTY_ID']):
                    df['PARTY_ID_num'] = df['PARTY_ID']
                # 方法2：如果PARTY_ID是分类或字符串，使用One-Hot编码或Label编码
                else:
                    label_encoder = LabelEncoder()
                    df['PARTY_ID_num'] = label_encoder.fit_transform(df['PARTY_ID'])
                    # logger.info(f"PARTY_ID已转换为数值特征PARTY_ID_num，共{len(label_encoder.classes_)}个唯一值")
            except Exception as e:
                # 如果转换失败，创建一个常数列作为替代
                # logger.warning(f"PARTY_ID转换为数值特征失败: {e}，使用常数列代替")
                df['PARTY_ID_num'] = 1
            
            # 删除原始PARTY_ID列
            df = df.drop(columns=['PARTY_ID'])
            # logger.info("原始PARTY_ID列已删除，添加了PARTY_ID_num数值特征列")

        # 设置日期索引
        if date_column and date_column in df.columns and df.index.name != date_column:
            df = df.set_index(date_column)
        
        # 过滤日期范围
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # 添加时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # 调用修改后的 enhance_peak_features 进行特征增强
        df = self.enhance_peak_features(
            df, 
            interval=interval,
            peak_hours=peak_hours,
            valley_hours=valley_hours, 
            peak_weight=peak_weight,
            valley_weight=valley_weight,
            value_column=value_column
        )

        # 添加滞后特征，使用传入的值列名(预先填充，避免NaN警告)
        if len(df) > 0:  # 确保有数据
            # 计算每个滞后期的点数
            lag_1_points = 1
            lag_4_points = int(60/interval)  # 1小时前
            lag_24_points = int(1440/interval)  # 1天前
            lag_48_points = int(2880/interval)  # 2天前
            lag_168_points = int(672*15/interval)  # 1周前
            
            # 创建临时值系列，包括滞后特征计算需要的前置时间
            # 计算最大滞后期
            max_lag = max(lag_1_points, lag_4_points, lag_24_points, lag_48_points, lag_168_points)
            
            # 为每个滞后期静默填充值(通过复制首值)
            temp_series = df[value_column].copy()
            
            # 创建滞后特征并填充
            df[f'{value_column}_lag_1'] = temp_series.shift(lag_1_points)
            df[f'{value_column}_lag_4'] = temp_series.shift(lag_4_points)
            df[f'{value_column}_lag_24'] = temp_series.shift(lag_24_points)
            df[f'{value_column}_lag_48'] = temp_series.shift(lag_48_points)
            df[f'{value_column}_lag_168'] = temp_series.shift(lag_168_points)
            
            # 立即填充NaN值，避免后续警告
            df = df.bfill().ffill()

        # 添加时间段类别特征
        # 将一天分为早高峰(6-10)、日间(10-16)、晚高峰(16-20)、夜间(20-6)
        time_period = pd.Series(index=df.index, dtype='int')
        time_period[(df['hour'] >= 6) & (df['hour'] < 10)] = 1  # 早高峰
        time_period[(df['hour'] >= 10) & (df['hour'] < 16)] = 2  # 日间
        time_period[(df['hour'] >= 16) & (df['hour'] < 20)] = 3  # 晚高峰
        time_period[(df['hour'] >= 20) | (df['hour'] < 6)] = 4   # 夜间
        
        # 通过独热编码添加时间段特征
        for i in range(1, 5):
            df[f'time_period_{i}'] = (time_period == i).astype(int)

        # 对字符串列进行编码，但排除已处理的PARTY_ID
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                # 使用独热编码
                one_hot = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, one_hot], axis=1)
                df.drop(columns=[col], inplace=True)

        # Final type conversion and NaN check
        # print("--- DEBUG: Applying final type conversion and fillna ---")
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in df.columns:
            if col not in numeric_cols:
                # Attempt conversion for non-numeric columns identified previously
                # Coerce errors will turn unconvertible values into NaN
                # Specifically handle boolean columns first, converting to int (0/1)
                if df[col].dtype == bool or all(isinstance(x, bool) for x in df[col].unique() if pd.notna(x)):
                    # print(f"  - Converting boolean column '{col}' to int.")
                    df[col] = df[col].astype(int)
                else:
                    # print(f"  - Attempting numeric conversion for object column '{col}'.")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Check if coercion created NaNs
            #         if df[col].isnull().any():
            #              print(f"    - NaNs created during numeric conversion for column '{col}'.")
            # elif df[col].isnull().any():
            #     # Also check existing numeric columns for NaNs that might have slipped through
            #      print(f"  - Warning: Existing numeric column '{col}' contains NaNs before final fillna.")

        # Fill any NaNs potentially introduced by 'coerce' or missed earlier
        # Using bfill().ffill() is generally robust
        # print("  - Applying final fillna (bfill then ffill)...")
        df = df.bfill().ffill()
        
        # # Check for NaNs *after* filling
        # if df.isnull().any().any():
        #      print("  - Warning: NaNs still present after final fillna!")
        #      print(df.isnull().sum()[df.isnull().sum() > 0])

        # # Final explicit cast to ensure float32
        # print("  - Attempting final cast to np.float32...")
        try:
            # Identify columns that are still not numeric after conversion attempts
            non_numeric_final = df.select_dtypes(exclude=np.number).columns
            if len(non_numeric_final) > 0:
                #  print(f"  - Warning: Columns {list(non_numeric_final)} could not be fully converted to numeric before casting.")
                 # Drop these problematic columns before casting? Or raise error?
                 # Option: Drop them
                 # df = df.drop(columns=non_numeric_final)
                 # print(f"    - Dropped non-numeric columns: {list(non_numeric_final)}")
                 # Option: Raise error
                 raise TypeError(f"Columns {list(non_numeric_final)} remain non-numeric before final cast.")
            
            df = df.astype(np.float32)
            # print("  - DataFrame successfully cast to np.float32.")
        except Exception as e:
            # print(f"  - ERROR: Could not cast entire DataFrame to np.float32: {e}")
            # print("    Current dtypes:")
            # print(df.dtypes)
            # Re-raise the exception to halt execution as this is critical
            raise e

        return df

    def apply_scaling(self, X, y=None, inverse=False):
        """
        应用标准化或反标准化
        
        参数:
        X: 特征数据
        y: 目标数据 (可选)
        inverse (bool): 是否执行反标准化
        
        返回:
        tuple: 处理后的X, 处理后的y (如果提供)
        """
        if 'X' not in self.scalers or 'y' not in self.scalers:
            raise ValueError("请先调用fit_scalers()以初始化标准化器")
        
        # 保存原始形状以便恢复
        X_shape = X.shape
        
        # 对X进行处理
        X_reshaped = X.reshape(X.shape[0], -1)
        if inverse:
            X_processed = self.scalers['X'].inverse_transform(X_reshaped)
        else:
            X_processed = self.scalers['X'].transform(X_reshaped)
        
        # 恢复原始形状
        X_processed = X_processed.reshape(X_shape)
        
        # 如果提供了y，也进行处理
        y_processed = None
        if y is not None:
            # 确保y有正确的形状
            y_is_1d = len(y.shape) == 1
            y_shaped = y.reshape(-1, 1) if y_is_1d else y
            
            if inverse:
                y_processed = self.scalers['y'].inverse_transform(y_shaped)
            else:
                y_processed = self.scalers['y'].transform(y_shaped)
            
            # 如果原始y是1D，转回1D
            if y_is_1d:
                y_processed = y_processed.flatten()
        
        return (X_processed, y_processed) if y is not None else X_processed

    def build_dataset_sequence_from_df(self, df, seq_length=None, include_time_features=True, 
                                      date_column='datetime', value_column='load',
                                      start_date=None, end_date=None, interval=15, pred_horizon=1,
                                      peak_hours=(8, 20), valley_hours=(0, 6), 
                                      peak_weight=2.0, valley_weight=1.5):
        """
        从DataFrame构建序列数据集，整合了高峰感知特征，适用于包含天气数据的时间序列预测模型
        
        参数:
        df (DataFrame): 包含时间和值数据的DataFrame
        seq_length (int): 输入序列长度，若为None则使用实例默认值
        include_time_features (bool): 是否包含时间特征
        date_column (str): 日期列名
        value_column (str): 值列名
        start_date: 开始日期
        end_date: 结束日期
        interval (int): 数据间隔（分钟）
        pred_horizon (int): 预测时间步长
        peak_hours (tuple): 高峰时段的开始和结束小时 (如 (8, 20))
        valley_hours (tuple): 低谷时段的开始和结束小时 (如 (0, 6))
        peak_weight (float): 高峰时段的权重倍数
        valley_weight (float): 低谷时段的权重倍数
        
        返回:
        tuple: (X, y) X为特征序列，y为目标值
        """
        # 使用实例默认值，如果未提供seq_length
        if seq_length is None:
            seq_length = self.seq_length
        
        # 确保日期列是索引
        if date_column in df.columns:
            df = df.set_index(date_column)
        
        # 过滤日期范围
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # 不直接复制原始数据，而是使用增强的数据处理策略
        # 区分是否需要添加时间特征
        if include_time_features:
            # 使用高峰感知特征工程构建数据集
            data_df = self.build_dataset_with_peak_awareness(
                df=df,
                date_column=None,  # 已经设置为索引
                value_column=value_column,
                peak_hours=peak_hours,
                valley_hours=valley_hours,
                peak_weight=peak_weight,
                valley_weight=valley_weight,
                start_date=None,  # 已经过滤过
                end_date=None     # 已经过滤过
            )
        else:
            # 如果不需要时间特征，则仅处理非数值列
            data_df = df.copy()
            
            # 处理非数值列，包括PARTY_ID
            categorical_columns = data_df.select_dtypes(include=['object']).columns.tolist()
            if 'PARTY_ID' in categorical_columns:
                # 将PARTY_ID转换为数值特征
                try:
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    data_df['PARTY_ID_num'] = label_encoder.fit_transform(data_df['PARTY_ID'])
                    data_df = data_df.drop(columns=['PARTY_ID'])
                except Exception as e:
                    print(f"PARTY_ID转换为数值特征失败: {e}，使用常数列代替")
                    data_df['PARTY_ID_num'] = 1
                    data_df = data_df.drop(columns=['PARTY_ID'])
        
        # 确保所有列都是数值类型
        for col in data_df.columns:
            if not pd.api.types.is_numeric_dtype(data_df[col]):
                try:
                    data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
                except Exception as e:
                    print(f"列 {col} 无法转换为数值类型: {e}，将其删除")
                    data_df = data_df.drop(columns=[col])
        
        # 填充缺失值
        data_df = data_df.bfill().ffill()
        
        # 确保所有数据都是float32类型，这对PyTorch很重要
        data_df = data_df.astype(np.float32)
        
        # 提取目标变量和特征
        y_values = data_df[value_column].values
        X_values = data_df.drop(columns=[value_column]).values
        
        # 创建特征和标签序列
        X, y = [], []
        for i in range(len(y_values) - seq_length - pred_horizon + 1):
            X.append(X_values[i:i + seq_length])
            y.append(y_values[i + seq_length:i + seq_length + pred_horizon])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # 如果是单步预测，去掉最后一个维度
        if pred_horizon == 1:
            y = y.reshape(-1)
        
        print(f"生成数据集: X形状={X.shape}, y形状={y.shape}")
        
        return X, y

    def prepare_weather_data_with_peak_awareness(self, ts_data, test_ratio=0.2, 
                                      peak_hours=(8, 20), valley_hours=(0, 6),
                                      peak_weight=2.0, valley_weight=1.5, 
                                      start_date=None, end_date=None, value_column='load'):
        """
        使用高峰和低谷感知特征准备包含天气数据的训练和验证数据集

        参数:
        ts_data (DataFrame): 包含负荷和天气数据的时间序列格式数据
        test_ratio (float): 测试集比例
        peak_hours (tuple): 高峰时段的开始和结束小时
        valley_hours (tuple): 低谷时段的开始和结束小时
        peak_weight (float): 高峰时段的权重倍数
        valley_weight (float): 低谷时段的权重倍数
        start_date (str): 数据集的开始日期
        end_date (str): 数据集的结束日期
        value_column (str): 值列的名称，默认为'load'

        返回:
        tuple: X_train, y_train, X_val, y_val
        """
        # 过滤日期范围
        filtered_data = ts_data.copy()
        if start_date:
            filtered_data = filtered_data[filtered_data.index >= pd.to_datetime(start_date)]
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= pd.to_datetime(end_date)]
            
        # 识别并保留天气特征
        # 假设除了以下列表之外的所有列都是天气特征
        non_weather_columns = [
            value_column, 'hour', 'day', 'month', 'weekday', 'is_weekend', 
            'datetime', 'date', 'PARTY_ID', 'timestamp'
        ]
        all_columns = filtered_data.columns.tolist()
        weather_features = [col for col in all_columns if col not in non_weather_columns]
        print(f"识别到 {len(weather_features)} 个天气特征: {weather_features}")

        # 调用 build_dataset_with_peak_awareness 增强数据
        enhanced_data = self.build_dataset_with_peak_awareness(
            df=filtered_data,
            date_column='datetime' if 'datetime' in filtered_data.columns else None,
            value_column=value_column,
            peak_hours=peak_hours,
            valley_hours=valley_hours,
            peak_weight=peak_weight,
            valley_weight=valley_weight,
            start_date=None,  # 前面已经过滤过
            end_date=None     # 前面已经过滤过
        )

        # 提取目标变量和特征
        y_values = enhanced_data[value_column].values
        X_values = enhanced_data.drop(columns=[value_column]).values

        # 创建特征和标签序列
        X, y = [], []
        for i in range(len(y_values) - self.seq_length - self.pred_horizon + 1):
            X.append(X_values[i:i + self.seq_length])
            y.append(y_values[i + self.seq_length:i + self.seq_length + self.pred_horizon])

        X = np.array(X)
        y = np.array(y).reshape(-1, self.pred_horizon)

        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 如果是单步预测，去掉最后一个维度
        if self.pred_horizon == 1:
            y_train = y_train.flatten()
            y_val = y_val.flatten()

        print(f"生成数据集: X_train形状={X_train.shape}, y_train形状={y_train.shape}")
        print(f"生成数据集: X_val形状={X_val.shape}, y_val形状={y_val.shape}")
        
        return X_train, y_train, X_val, y_val