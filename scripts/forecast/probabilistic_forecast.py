# data/probabilistic_forecast.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm

# Import necessary components
from models.torch_models import ProbabilisticConvTransformer
from utils.scaler_manager import ScalerManager
from data.dataset_builder import DatasetBuilder
from utils.interval_forecast_utils import DataPipeline, create_prediction_intervals, plot_interval_forecast
from models.torch_models import IntervalPeakAwareConvTransformer
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 新增：用于区间预测的类
class IntervalPredictionManager:
    """
    基于多模型集成和误差分布的预测区间生成器
    
    该方法通过以下步骤生成预测区间：
    1. 对有限的历史数据进行分组，训练多个确定性模型
    2. 使用交叉验证生成预测误差序列
    3. 根据置信水平和误差分布，计算最优参数来生成最小宽度的预测区间
    """
    
    def __init__(self, data_path, dataset_id='上海', forecast_type='load', 
                 n_models=5, cv_folds=5, model_dir='models/interval_prediction'):
        """
        初始化区间预测管理器
        
        Args:
            data_path (str): 时间序列数据路径
            dataset_id (str): 数据集标识
            forecast_type (str): 预测类型 ('load', 'pv', 'wind')
            n_models (int): 要训练的模型数量
            cv_folds (int): 交叉验证折数
            model_dir (str): 模型和误差序列存储目录
        """
        self.data_path = data_path
        self.dataset_id = dataset_id
        self.forecast_type = forecast_type
        self.n_models = n_models
        self.cv_folds = cv_folds
        
        # 设置目录
        self.base_dir = f"{model_dir}/{forecast_type}/{dataset_id}"
        self.models_dir = f"{self.base_dir}/models"
        self.errors_dir = f"{self.base_dir}/errors"
        self.results_dir = f"{self.base_dir}/results"
        
        # 创建必要的目录
        for dir_path in [self.models_dir, self.errors_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 模型列表和误差序列
        self.models = []
        self.error_sequences = {}
        self.is_trained = False
    
    def train_models(self, train_start_date, train_end_date, test_ratio=0.2, 
                    seq_length=96, interval_minutes=15, epochs=50, batch_size=32, retrain=False):
        """
        训练多个确定性模型并生成预测误差序列
        
        Args:
            train_start_date (str): 训练数据开始日期 'YYYY-MM-DD'
            train_end_date (str): 训练数据结束日期 'YYYY-MM-DD'
            test_ratio (float): 测试集比例
            seq_length (int): 输入序列长度
            interval_minutes (int): 数据间隔（分钟）
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            retrain (bool): 是否重新训练已有模型
        
        Returns:
            bool: 训练是否成功
        """
        # 检查是否已训练且不需要重新训练
        models_exist = all(os.path.exists(f"{self.models_dir}/model_{i}") for i in range(self.n_models))
        errors_exist = os.path.exists(f"{self.errors_dir}/error_sequences.pkl")
        
        if models_exist and errors_exist and not retrain:
            print(f"发现已训练的模型和误差序列。加载现有数据...")
            self._load_models_and_errors()
            return True
        
        # 加载数据
        print(f"从 {self.data_path} 加载数据...")
        try:
            ts_data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            value_column = self.forecast_type
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False
        
        # 筛选训练时间范围内的数据
        start_dt = pd.to_datetime(train_start_date)
        end_dt = pd.to_datetime(train_end_date)
        train_data = ts_data[(ts_data.index >= start_dt) & (ts_data.index <= end_dt)].copy()
        
        if len(train_data) < seq_length * 2:
            print(f"错误: 训练数据不足，需要至少 {seq_length * 2} 个点，但只有 {len(train_data)} 个点")
            return False
        
        # 准备数据集生成器
        dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1)
        
        # 特征工程
        print("进行特征工程...")
        try:
            enhanced_data = dataset_builder.build_dataset_with_peak_awareness(
                df=train_data, 
                date_column=None, 
                value_column=value_column,
                interval=interval_minutes,
                peak_hours=(7, 22),
                valley_hours=(0, 6)
            )
        except Exception as e:
            print(f"特征工程时出错: {e}")
            return False
        
        # 准备特征和目标
        feature_columns = [col for col in enhanced_data.columns if col != value_column]
        X = enhanced_data[feature_columns].values
        y = enhanced_data[value_column].values
        
        # 创建时间索引划分器，确保时间上的连续性
        total_samples = len(enhanced_data)
        split_idx = int(total_samples * (1 - test_ratio))
        
        X_all = X[:split_idx]
        y_all = y[:split_idx]
        
        # 用于测试误差计算的数据
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # 将数据分成n_models组，确保每组数据量相近但有所不同
        group_size = len(X_all) // self.n_models
        model_groups = []
        
        # 创建数据分组，每组数据有一定的重叠
        for i in range(self.n_models):
            start_idx = max(0, i * group_size // 2)
            end_idx = min(len(X_all), start_idx + group_size)
            model_groups.append((X_all[start_idx:end_idx], y_all[start_idx:end_idx]))
        
        # 训练模型并收集误差
        self.models = []
        all_errors = []
        
        print(f"开始训练 {self.n_models} 个模型并生成误差序列...")
        for i, (X_group, y_group) in enumerate(model_groups):
            print(f"\n训练模型 {i+1}/{self.n_models}...")
            model_save_dir = f"{self.models_dir}/model_{i}"
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 使用K折交叉验证生成误差
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            fold_errors = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_group)):
                print(f"  训练交叉验证折 {fold+1}/{self.cv_folds}...")
                X_fold_train, X_fold_val = X_group[train_idx], X_group[val_idx]
                y_fold_train, y_fold_val = y_group[train_idx], y_group[val_idx]
                
                # 准备标准化器
                scaler_dir = f"{model_save_dir}/fold_{fold}/scalers"
                os.makedirs(scaler_dir, exist_ok=True)
                scaler_manager = ScalerManager()
                
                # 标准化数据
                X_fold_train_scaled = scaler_manager.fit_transform('X', X_fold_train)
                y_fold_train_scaled = scaler_manager.fit_transform('y', y_fold_train.reshape(-1, 1)).flatten()
                X_fold_val_scaled = scaler_manager.transform('X', X_fold_val)
                y_fold_val_scaled = scaler_manager.transform('y', y_fold_val.reshape(-1, 1)).flatten()
                
                # 保存标准化器
                os.makedirs(scaler_dir, exist_ok=True)
                scaler_manager.save_scaler('X', scaler_manager.get_scaler('X'))
                scaler_manager.save_scaler('y', scaler_manager.get_scaler('y'))
                
                # 直接使用原始特征，不进行复杂的重塑
                print(f"  训练数据形状: {X_fold_train_scaled.shape}")
                print(f"  验证数据形状: {X_fold_val_scaled.shape}")
                
                # 初始化模型
                model = ProbabilisticConvTransformer(
                    input_shape=X_fold_train_scaled.shape[1:] if len(X_fold_train_scaled.shape) > 1 else (X_fold_train_scaled.shape[0], 1),
                    quantiles=[0.5], # 只使用中位数预测
                    seq_length=seq_length,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=0.001,
                    patience=10
                )
                
                # 训练模型
                model.train_probabilistic(
                    X_fold_train_scaled,
                    y_fold_train_scaled.reshape(-1, 1),
                    X_fold_val_scaled,
                    y_fold_val_scaled.reshape(-1, 1),
                    epochs=epochs,
                    batch_size=batch_size,
                    save_dir=f"{model_save_dir}/fold_{fold}"
                )
                
                # 计算验证集误差
                val_preds = model.predict_probabilistic(
                    X_fold_val_scaled
                )
                
                # 提取中位数预测并反标准化
                val_preds_p50 = val_preds['p50']
                val_preds_p50_unscaled = scaler_manager.inverse_transform('y', val_preds_p50.reshape(-1, 1)).flatten()
                
                # 计算误差 (预测值 - 实际值)
                errors = val_preds_p50_unscaled - y_fold_val
                fold_errors.extend(errors.tolist())
            
            # 保存所有折的误差
            all_errors.extend(fold_errors)
            
            # 在所有数据上训练最终模型
            print(f"  训练最终模型 {i+1}...")
            
            # 标准化全部数据
            scaler_manager = ScalerManager()
            X_group_scaled = scaler_manager.fit_transform('X', X_group)
            y_group_scaled = scaler_manager.fit_transform('y', y_group.reshape(-1, 1)).flatten()
            
            # 保存标准化器
            scaler_dir = f"{model_save_dir}/scalers"
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_manager.save_scaler('X', scaler_manager.get_scaler('X'))
            scaler_manager.save_scaler('y', scaler_manager.get_scaler('y'))
            
            # 直接使用原始特征，不进行复杂的重塑
            print(f"  最终模型的训练数据形状: {X_group_scaled.shape}")
            
            # 初始化模型
            model = ProbabilisticConvTransformer(
                input_shape=X_group_scaled.shape[1:] if len(X_group_scaled.shape) > 1 else (X_group_scaled.shape[0], 1),
                quantiles=[0.5], # 只使用中位数预测
                seq_length=seq_length,
                epochs=epochs,
                batch_size=batch_size,
                lr=0.001,
                patience=10
            )
            
            # 训练模型
            model.train_probabilistic(
                X_group_scaled,
                y_group_scaled.reshape(-1, 1),
                X_group_scaled,  # 使用相同数据作为验证（因为是最终模型）
                y_group_scaled.reshape(-1, 1),
                epochs=epochs//2,  # 减少epochs防止过拟合
                batch_size=batch_size,
                save_dir=model_save_dir
            )
            
            # 保存最终模型
            self.models.append({
                'model': model,
                'scaler_dir': f"{model_save_dir}/scalers"
            })
        
        # 对测试集进行预测
        print("评估测试集并生成最终误差序列...")
        test_errors = []
        
        for i, model_info in enumerate(self.models):
            model = model_info['model']
            scaler_manager = ScalerManager(scaler_path=model_info['scaler_dir'])
            
            # 标准化测试数据
            X_test_scaled = scaler_manager.transform('X', X_test)
            
            # 直接使用原始特征，不进行复杂的重塑
            print(f"  测试数据形状: {X_test_scaled.shape}")
            
            # 预测
            test_preds = model.predict_probabilistic(X_test_scaled)
            
            # 提取中位数预测并反标准化
            test_preds_p50 = test_preds['p50']
            test_preds_p50_unscaled = scaler_manager.inverse_transform('y', test_preds_p50.reshape(-1, 1)).flatten()
            
            # 计算误差 (预测值 - 实际值)
            errors = test_preds_p50_unscaled - y_test
            test_errors.extend(errors.tolist())
        
        # 合并所有误差序列并排序
        all_errors.extend(test_errors)
        sorted_errors = np.sort(all_errors)
        
        # 保存误差序列
        self.error_sequences = {
            'all': sorted_errors,
            'cv': np.sort(all_errors[:-len(test_errors)]),
            'test': np.sort(test_errors)
        }
        
        # 保存误差序列
        with open(f"{self.errors_dir}/error_sequences.pkl", 'wb') as f:
            pickle.dump(self.error_sequences, f)
        
        print(f"区间预测模型和误差序列训练完成，保存到 {self.base_dir}")
        self.is_trained = True
        return True
    
    def _load_models_and_errors(self):
        """加载已训练的模型和误差序列"""
        # 加载误差序列
        with open(f"{self.errors_dir}/error_sequences.pkl", 'rb') as f:
            self.error_sequences = pickle.load(f)
        
        # 加载模型
        self.models = []
        for i in range(self.n_models):
            model_dir = f"{self.models_dir}/model_{i}"
            try:
                model = ProbabilisticConvTransformer.load(model_dir)
                self.models.append({
                    'model': model,
                    'scaler_dir': f"{model_dir}/scalers"
                })
            except Exception as e:
                print(f"加载模型 {i} 时出错: {e}")
        
        self.is_trained = True
        print(f"已加载 {len(self.models)} 个模型和误差序列")
    
    def calculate_optimal_beta(self, alpha=0.1, method='min_width'):
        """
        计算最优beta参数以生成最小宽度的预测区间
        
        Args:
            alpha (float): 置信水平 (1-alpha)
            method (str): 优化方法，'min_width'或'balanced'
        
        Returns:
            float: 最优beta值
        """
        if not self.is_trained or not self.error_sequences:
            raise ValueError("请先训练模型并生成误差序列")
        
        errors = self.error_sequences['all']
        n = len(errors)
        
        if method == 'min_width':
            # 找到使区间宽度最小的beta值
            min_width = float('inf')
            optimal_beta = 0
            
            for beta_idx in range(int(n * alpha)):
                lower_bound = errors[beta_idx]
                upper_bound = errors[int(n * (1 - alpha + beta_idx/n))]
                width = upper_bound - lower_bound
                
                if width < min_width:
                    min_width = width
                    optimal_beta = beta_idx / n
        
        elif method == 'balanced':
            # 平衡两侧概率，使区间居中
            optimal_beta = alpha / 2
        
        else:
            raise ValueError(f"不支持的方法: {method}，请使用 'min_width' 或 'balanced'")
        
        return optimal_beta
    
    def predict_with_interval(self, forecast_date, confidence_level=0.9, 
                             method='min_width', interval_minutes=15):
        """
        执行区间预测
        
        Args:
            forecast_date (str): 预测日期 'YYYY-MM-DD'
            confidence_level (float): 置信水平 (0-1)
            method (str): 计算最优区间的方法，'min_width'或'balanced'
            interval_minutes (int): 数据间隔（分钟）
        
        Returns:
            DataFrame: 包含预测结果和区间的DataFrame
        """
        if not self.is_trained:
            raise ValueError("请先训练模型并生成误差序列")
        
        if len(self.models) == 0:
            raise ValueError("没有可用的模型")
        
        # 加载数据
        print(f"从 {self.data_path} 加载数据...")
        try:
            ts_data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            value_column = self.forecast_type
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
        
        # 预处理
        forecast_day = pd.to_datetime(forecast_date)
        seq_length = self.models[0]['model'].config.get('seq_length', 96)
        
        # 获取历史数据作为输入
        historical_end_dt = forecast_day
        historical_start_dt = historical_end_dt - timedelta(minutes=seq_length * interval_minutes)
        historical_data = ts_data[(ts_data.index >= historical_start_dt) & (ts_data.index < historical_end_dt)].copy()
        
        if len(historical_data) < seq_length:
            raise ValueError(f"历史数据不足 ({len(historical_data)} < {seq_length})")
        
        if len(historical_data) > seq_length:
            historical_data = historical_data.iloc[-seq_length:]
        
        # 特征工程
        dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1)
        
        enhanced_history = dataset_builder.build_dataset_with_peak_awareness(
            df=historical_data, 
            date_column=None, 
            value_column=value_column,
            interval=interval_minutes,
            peak_hours=(7, 22),
            valley_hours=(0, 6)
        )
        
        # 准备模型输入
        feature_columns = [col for col in enhanced_history.columns if col != value_column]
        X = enhanced_history[feature_columns].values
        X_input = X.reshape(1, seq_length, -1)
        
        # 使用多个模型进行预测
        model_predictions = []
        
        for model_info in self.models:
            model = model_info['model']
            scaler_manager = ScalerManager(scaler_path=model_info['scaler_dir'])
            
            # 标准化输入
            X_scaled = scaler_manager.transform('X', X_input.reshape(1, -1))
            
            # 直接使用原始特征进行预测
            print(f"  预测输入数据形状: {X_scaled.shape}")
            
            # 预测
            predictions = model.predict_probabilistic(X_scaled)
            
            # 提取中位数预测并反标准化
            p50_scaled = predictions['p50']
            p50 = scaler_manager.inverse_transform('y', p50_scaled.reshape(-1, 1)).flatten()
            
            model_predictions.append(p50[0])
        
        # 计算集成预测结果（平均值）
        ensemble_prediction = np.mean(model_predictions)
        
        # 计算最优beta
        alpha = 1 - confidence_level
        beta = self.calculate_optimal_beta(alpha, method)
        
        # 使用误差序列和最优beta生成预测区间
        errors = self.error_sequences['all']
        n = len(errors)
        
        lower_bound_idx = int(n * beta)
        upper_bound_idx = int(n * (1 - alpha + beta))
        
        lower_bound_error = errors[lower_bound_idx]
        upper_bound_error = errors[upper_bound_idx]
        
        # 计算预测区间
        lower_bound = ensemble_prediction + lower_bound_error
        upper_bound = ensemble_prediction + upper_bound_error
        
        # 准备多个时间步的预测
        num_predictions = int(24 * 60 / interval_minutes)
        forecast_start = pd.Timestamp(f"{forecast_date} 00:00:00")
        forecast_times = pd.date_range(start=forecast_start, periods=num_predictions, freq=f'{interval_minutes}min')
        
        # 由于当前实现只进行单步预测，所以将结果复制到多个时间步
        # 在实际应用中，可以实现递归多步预测或多步模型
        ensemble_predictions = np.full(num_predictions, ensemble_prediction)
        lower_bounds = np.full(num_predictions, lower_bound)
        upper_bounds = np.full(num_predictions, upper_bound)
        
        # 创建结果DataFrame
        results_data = {
            'datetime': forecast_times,
            'prediction': ensemble_predictions,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'interval_width': upper_bounds - lower_bounds
        }
        
        results_df = pd.DataFrame(results_data)
        
        # 添加实际值（如果有）
        actual_start = forecast_start
        actual_end = forecast_times[-1]
        actual_data = ts_data[(ts_data.index >= actual_start) & (ts_data.index <= actual_end)].copy()
        
        if not actual_data.empty:
            results_df = results_df.set_index('datetime')
            actual_series = actual_data[value_column].reindex(results_df.index)
            results_df['actual'] = actual_series
            results_df = results_df.reset_index()
        
        # 可视化结果
        plt.figure(figsize=(12, 6))
        
        # 绘制预测区间
        plt.fill_between(results_df['datetime'], 
                         results_df['lower_bound'], 
                         results_df['upper_bound'], 
                         color='blue', alpha=0.2, 
                         label=f'{confidence_level*100}% 预测区间')
        
        # 绘制预测值
        plt.plot(results_df['datetime'], results_df['prediction'], 'b--', 
                 label='预测值', linewidth=1.5)
        
        # 绘制实际值（如果有）
        if 'actual' in results_df.columns and not results_df['actual'].isna().all():
            plt.plot(results_df['datetime'], results_df['actual'], 'k-', 
                    label='实际值', linewidth=2)
        
        plt.title(f"区间预测 ({self.forecast_type.upper()}) - {forecast_date} ({self.dataset_id})")
        plt.xlabel('时间')
        plt.ylabel(f'{self.forecast_type.capitalize()} (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表和结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"{self.results_dir}/interval_forecast_{forecast_date}_{timestamp}.png"
        csv_path = f"{self.results_dir}/interval_forecast_{forecast_date}_{timestamp}.csv"
        
        plt.savefig(plot_path, dpi=300)
        print(f"区间预测图表已保存至 {plot_path}")
        plt.close()
        
        results_df.to_csv(csv_path, index=False)
        print(f"区间预测结果已保存至 {csv_path}")
        
        return results_df

# 执行区间预测的函数
def perform_interval_forecast(data_path, forecast_date, dataset_id, forecast_type='load', 
                             confidence_level=0.9, retrain=False, historical_days=8):
    """
    执行区间预测并返回结果
    
    参数:
    - data_path: 数据文件路径
    - forecast_date: 预测日期
    - dataset_id: 数据集ID（省份）
    - forecast_type: 预测类型 (load/pv/wind)
    - confidence_level: 置信水平 (0.0-1.0)
    - retrain: 是否重新训练模型
    - historical_days: 用于预测的历史数据天数
    
    返回:
    - DataFrame包含预测结果
    """
    try:
        # 加载数据
        data = pd.read_csv(data_path)
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index('datetime', inplace=True)
        
        # 检查数据文件是否存在
        data_file = f'data/timeseries_{forecast_type}_{dataset_id}.csv'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
            
        # --- 修改：直接使用 ScalerManager 加载 --- 
        scaler_dir = f"models/scalers/convtrans_peak/{forecast_type}/{dataset_id}" # 与训练保存路径一致
        scaler_manager = ScalerManager(scaler_path=scaler_dir)
        if not scaler_manager.has_scaler('X') or not scaler_manager.has_scaler('y'):
             # 尝试显式加载，如果不存在会报错
             try:
                 scaler_manager.load_scaler('X')
                 scaler_manager.load_scaler('y')
                 if not scaler_manager.has_scaler('X') or not scaler_manager.has_scaler('y'):
                    raise FileNotFoundError(f"标准化器文件缺失于: {scaler_dir}")
             except Exception as load_err:
                 print(f"加载标准化器时显式出错: {load_err}")
                 raise FileNotFoundError(f"无法加载标准化器，请确保模型已训练并保存了缩放器于: {scaler_dir}")
        print(f"成功初始化并找到标准化器于: {scaler_dir}")
        # ---------------------------------------
        
        # 加载历史数据
        forecast_date_dt = datetime.strptime(forecast_date, '%Y-%m-%d')
        start_date_dt = forecast_date_dt - timedelta(days=historical_days)
        
        # 准备历史数据
        hist_data = data[(data.index >= start_date_dt) & (data.index < forecast_date_dt)]
        if hist_data.empty:
            raise ValueError(f"历史数据不足，无法进行区间预测")
        
        # ---- 修改：移除 DataPipeline 特征准备，手动准备 ----
        # 准备特征 (与训练时一致，使用 interval_forecast_utils 中的函数)
        from utils.interval_forecast_utils import create_temporal_features, is_peak_hour, is_valley_hour, FEATURES_CONFIG
        
        # 添加时间特征
        hist_data_with_features = create_temporal_features(hist_data)
        # 添加峰谷标记 (假设函数可用)
        hist_data_with_features['is_peak'] = [is_peak_hour(h) for h in hist_data_with_features.index.hour]
        hist_data_with_features['is_valley'] = [is_valley_hour(h) for h in hist_data_with_features.index.hour]
        
        # 添加滞后特征
        target_col = forecast_type
        # --- 修改：直接使用默认的滞后列表 --- 
        lags_to_use = [1, 4, 24, 48, 168] # 使用默认的滞后值
        # (FEATURES_CONFIG 用于后续选择最终特征列)
        # -------------------------------------
        for lag in lags_to_use:
             lag_name = f"{target_col}_lag_{lag}"
             hist_data_with_features[lag_name] = hist_data_with_features[target_col].shift(lag)
        
        # 处理NaN
        hist_data_with_features = hist_data_with_features.dropna()
        if hist_data_with_features.empty:
            raise ValueError("添加特征后无有效历史数据")
        
        # 提取X特征
        feature_cols = FEATURES_CONFIG.get(forecast_type, [])
        # 确保所有特征列都存在
        missing_cols = [col for col in feature_cols if col not in hist_data_with_features.columns]
        if missing_cols:
            raise ValueError(f"历史数据中缺少以下特征列: {missing_cols}")
            
        X_hist = hist_data_with_features[feature_cols].values
        # 提取 is_peak 和 is_valley
        is_peak_hist = hist_data_with_features['is_peak'].values
        is_valley_hist = hist_data_with_features['is_valley'].values
        # -------------------------------------------------
        
        # 加载模型
        model_dir = f"models/convtrans_peak/{forecast_type}/{dataset_id}"
        model = IntervalPeakAwareConvTransformer.load(model_dir)
        
        # ---- 修改：使用加载的 ScalerManager 进行缩放 ---- 
        # 缩放特征
        X_hist_scaled = scaler_manager.transform('X', X_hist) 
        # -------------------------------------------
        
        # 进行预测
        print("执行点预测...")
        # --- 预测时需要提供 (batch, seq_len, features) 形状 --- 
        # 注意：这里的逻辑假设模型直接预测整个序列，而不是滚动预测
        # 这可能需要调整，取决于模型的 predict 方法如何处理输入
        # 如果模型需要 [batch, seq_len, features]，则需要调整 X_hist_scaled 的形状
        # 假设模型.predict 可以处理 (num_samples, features) 的输入
        predictions = model.predict(X_hist_scaled) 
        
        # 根据时段特征获取误差分布
        print("创建预测区间...")
        intervals = model.predict_interval(
            X_hist_scaled, 
            is_peak=is_peak_hist,
            is_valley=is_valley_hist
        )
        # --------------------------------------------------
        
        # ---- 修改：使用加载的 ScalerManager 进行反缩放 ----
        # 反缩放预测结果
        for q, pred_values in intervals.items():
            intervals[q] = scaler_manager.inverse_transform('y', pred_values)
        # -------------------------------------------------
            
        # 创建结果DataFrame
        # 使用历史数据的索引，因为我们是对历史数据进行的预测
        timestamps = hist_data_with_features.index
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(index=timestamps)
        results_df['datetime'] = timestamps
        
        # 添加各分位数的预测结果
        for q, pred_values in intervals.items():
             results_df[q] = pred_values
             
        # 重命名中位数预测为 'prediction'
        if 'p50' in results_df.columns:
            results_df.rename(columns={'p50': 'prediction'}, inplace=True)
        else:
             # 如果没有p50，选择一个接近的作为点预测，或报错
             print("警告: 预测结果中缺少 p50 (中位数)")
             # 可以选择 p25 或 p75，或者简单地使用第一个分位数
             first_q_key = list(intervals.keys())[0]
             results_df['prediction'] = results_df[first_q_key] 

        # 添加峰谷标记
        results_df['is_peak'] = is_peak_hist
        results_df['is_valley'] = is_valley_hist
        
        # 根据置信水平计算区间边界
        lower_quantile = (1 - confidence_level) / 2
        upper_quantile = 1 - lower_quantile
        
        # 找到最接近的可用分位数
        available_quantiles_numeric = [float(q.replace('p', '')) / 100 for q in intervals.keys()]
        lower_q_numeric = min(available_quantiles_numeric, key=lambda x: abs(x - lower_quantile))
        upper_q_numeric = min(available_quantiles_numeric, key=lambda x: abs(x - upper_quantile))
        
        lower_q_key = f'p{int(lower_q_numeric * 100)}'
        upper_q_key = f'p{int(upper_q_numeric * 100)}'
        
        # 添加区间边界
        results_df['lower_bound'] = results_df[lower_q_key]
        results_df['upper_bound'] = results_df[upper_q_key]
        
        # 寻找对应日期的实际数据（如果有）
        # 注意：这里的逻辑需要调整，因为是对历史数据做的预测
        results_df['actual'] = hist_data_with_features[target_col].values
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        interval_results_dir = f"results/interval_prediction/{forecast_type}/{dataset_id}"
        os.makedirs(interval_results_dir, exist_ok=True)
        # --- 修改文件名以反映实际预测的日期 --- 
        hist_start_str = timestamps.min().strftime("%Y%m%d")
        hist_end_str = timestamps.max().strftime("%Y%m%d")
        results_path = f"{interval_results_dir}/interval_forecast_{hist_start_str}_{hist_end_str}_{timestamp}.csv"
        # -------------------------------------
        results_df.to_csv(results_path, index=False)
        print(f"区间预测结果已保存到 {results_path}")
        
        # 绘制并保存图表
        try:
            fig = plot_interval_forecast(
                results_df['datetime'],
                results_df['prediction'],
                np.column_stack((results_df['lower_bound'], results_df['upper_bound'])),
                actuals=results_df['actual'] if 'actual' in results_df.columns else None,
                title=f"{dataset_id} {forecast_type} 区间预测 ({forecast_date})"
            )
            
            # 保存图表
            fig_path = f"{interval_results_dir}/interval_forecast_{forecast_date}_{timestamp}.png"
            fig.savefig(fig_path)
            print(f"区间预测图表已保存到 {fig_path}")
        except Exception as e:
            print(f"保存图表时出错: {e}")
        
        return results_df
    
    except Exception as e:
        print(f"执行区间预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def perform_probabilistic_forecast(
    data_path,
    forecast_date,
    dataset_id="上海",
    forecast_type="load",
    quantiles=[0.1, 0.5, 0.9],
    interval_minutes=15, # Add interval here
    peak_hours=(7, 22), # Keep for feature engineering
    valley_hours=(0, 6)
):
    """
    执行概率预测 (例如日前预测)

    Args:
        data_path (str): 时间序列数据路径.
        forecast_date (str): 预测日期 'YYYY-MM-DD'.
        dataset_id (str): 数据集标识.
        forecast_type (str): 预测类型 ('load', 'pv', 'wind').
        quantiles (list): 需要预测的分位数.
        interval_minutes (int): 数据时间间隔（分钟）.
        peak_hours (tuple): 高峰时段（用于特征工程）.
        valley_hours (tuple): 低谷时段（用于特征工程）.

    Returns:
        DataFrame: 包含预测结果 (多个分位数) 和实际值的DataFrame.
    """
    print(f"\n=== 执行概率预测 ({forecast_type.upper()}): {forecast_date} ({dataset_id}) ===")
    print(f"Quantiles: {quantiles}")

    # 1. 设置目录和模型名称
    model_base_name = ProbabilisticConvTransformer.model_type # 'prob_convtrans'
    model_dir = f"models/{model_base_name}/{forecast_type}/{dataset_id}"
    scaler_dir = f"models/scalers/{model_base_name}/{forecast_type}/{dataset_id}"
    results_dir = f"results/probabilistic/{forecast_type}/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)

    # 2. 加载模型和标准化器
    try:
        print(f"加载概率模型从: {model_dir}")
        model = ProbabilisticConvTransformer.load(save_dir=model_dir)
        # Ensure the loaded model has the correct quantiles (or handle mismatch)
        if set(model.quantiles) != set(quantiles):
            print(f"警告: 请求的分位数 {quantiles} 与加载模型的 {model.quantiles} 不同。将使用模型的分位数。")
            quantiles = model.quantiles # Use the model's quantiles
            
        seq_length = model.config.get('seq_length', 96)
        print(f"从模型配置获取 seq_length: {seq_length}")
        
        print(f"加载标准化器从: {scaler_dir}")
        scaler_manager = ScalerManager(scaler_path=scaler_dir)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先训练对应的概率模型。")
        return None
    except Exception as e:
         print(f"加载模型或标准化器时出错: {e}")
         import traceback
         traceback.print_exc()
         return None

    # 3. 加载和准备输入数据 (类似日前预测)
    print("加载和准备输入数据...")
    ts_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    value_column = forecast_type

    try:
        forecast_day = pd.to_datetime(forecast_date)
        historical_end_dt = forecast_day
        historical_start_dt = historical_end_dt - timedelta(minutes=seq_length * interval_minutes)
        historical_data = ts_data[(ts_data.index >= historical_start_dt) & (ts_data.index < historical_end_dt)].copy()

        if len(historical_data) < seq_length:
            raise ValueError(f"历史数据不足 ({len(historical_data)} < {seq_length})")
        elif len(historical_data) > seq_length:
            historical_data = historical_data.iloc[-seq_length:]

        # 特征工程 (使用 DatasetBuilder)
        dataset_builder = DatasetBuilder(seq_length=seq_length, pred_horizon=1)
        # Pass peak/valley info for feature creation
        enhanced_history = dataset_builder.build_dataset_with_peak_awareness(
            df=historical_data, date_column=None, value_column=value_column,
            interval=interval_minutes, 
            peak_hours=peak_hours, valley_hours=valley_hours, 
            peak_weight=1, valley_weight=1 # Weights not needed for prediction features
        )

        feature_columns = [col for col in enhanced_history.columns if col != value_column]
        X = enhanced_history[feature_columns].values
        if X.shape[0] != seq_length:
            raise ValueError(f"特征提取后数据点数 ({X.shape[0]}) 与 seq_length ({seq_length}) 不匹配")

        X_input = X.reshape(1, seq_length, -1) # Shape: [1, seq_len, features]
        print(f"模型输入形状: {X_input.shape}")
        
        # 标准化输入
        X_scaled = scaler_manager.transform('X', X_input.reshape(1, -1))
        
    except ValueError as e:
        print(f"准备输入数据时出错: {e}")
        return None
    except Exception as e:
        print(f"准备输入数据时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 4. 执行概率预测
    print("执行概率预测...")
    try:
        # 使用模型的 predict_probabilistic 方法
        quantile_predictions_scaled = model.predict_probabilistic(X_scaled)
    except Exception as e:
         print(f"概率预测执行失败: {e}")
         import traceback
         traceback.print_exc()
         return None
         
    # 5. 反标准化预测结果
    print("反标准化预测结果...")
    quantile_predictions = {}
    try:
        for q_label, values_scaled in quantile_predictions_scaled.items():
            # values_scaled should be shape [num_predictions] (usually 1 for quantile models)
            # inverse_transform expects shape [n_samples, 1]
            values_unscaled = scaler_manager.inverse_transform('y', values_scaled.reshape(-1, 1)).flatten()
            quantile_predictions[q_label] = values_unscaled
            
    except Exception as e:
         print(f"反标准化失败: {e}")
         import traceback
         traceback.print_exc()
         return None

    # --- IMPORTANT: Handle Multi-Step Probabilistic Forecast --- 
    # The current structure predicts only ONE step ahead based on quantile loss training.
    # To get a full day's forecast, we need a rolling probabilistic method.
    # For simplicity now, we will assume the model *could* output multiple steps 
    # or we repeat the single prediction (less accurate).
    # Let's assume the model was trained to output 96 steps (though quantile loss makes this hard)
    # OR we implement a recursive probabilistic forecast (more complex). 
    
    # --- Placeholder: Assuming model outputs 96 steps (Needs Model Adjustment) --- 
    num_predictions = int(24 * 60 / interval_minutes)
    if len(next(iter(quantile_predictions.values()))) == 1 and num_predictions > 1:
         print("警告: 模型只输出单步概率预测。将重复该预测填充整天 (可能不准确)。")
         for q_label in quantile_predictions:
             single_pred = quantile_predictions[q_label][0]
             quantile_predictions[q_label] = np.full(num_predictions, single_pred)
    elif len(next(iter(quantile_predictions.values()))) != num_predictions:
        print(f"警告: 模型输出步数与预期 ({num_predictions}) 不匹配。将截断或填充。")
        current_len = len(next(iter(quantile_predictions.values())))
        for q_label in quantile_predictions:
            if current_len < num_predictions:
                last_val = quantile_predictions[q_label][-1]
                padding = np.full(num_predictions - current_len, last_val)
                quantile_predictions[q_label] = np.concatenate([quantile_predictions[q_label], padding])
            else:
                quantile_predictions[q_label] = quantile_predictions[q_label][:num_predictions]
    # -------------------------------------------------------------------

    # 6. 准备结果DataFrame
    forecast_start = pd.Timestamp(f"{forecast_date} 00:00:00")
    forecast_times = pd.date_range(start=forecast_start, periods=num_predictions, freq=f'{interval_minutes}min')
    
    results_data = {'datetime': forecast_times}
    results_data.update(quantile_predictions) # Add p10, p50, p90 etc.

    results_df = pd.DataFrame(results_data)
    
    # 添加实际值 (如果有)
    actual_start = forecast_start
    actual_end = forecast_times[-1]
    actual_data = ts_data[(ts_data.index >= actual_start) & (ts_data.index <= actual_end)].copy()
    if not actual_data.empty:
        results_df = results_df.set_index('datetime')
        # Use reindex and fillna for robust merging
        actual_series = actual_data[value_column].reindex(results_df.index)
        results_df['actual'] = actual_series
        results_df = results_df.reset_index()
        
    # 7. 可视化结果
    plt.figure(figsize=(12, 6))
    # Plot actual value if available
    if 'actual' in results_df.columns and not results_df['actual'].isna().all():
        plt.plot(results_df['datetime'], results_df['actual'], 'k-', label='Actual', linewidth=2)
        
    # Plot median (p50) forecast
    median_col = 'p50' # Assume p50 is always present
    if median_col in results_df.columns:
        plt.plot(results_df['datetime'], results_df[median_col], 'b--', label='Median Forecast (P50)', linewidth=1.5)
    
    # Plot prediction interval (e.g., P10-P90)
    lower_bound_col = 'p10'
    upper_bound_col = 'p90'
    if lower_bound_col in results_df.columns and upper_bound_col in results_df.columns:
        plt.fill_between(results_df['datetime'], 
                         results_df[lower_bound_col], 
                         results_df[upper_bound_col], 
                         color='blue', alpha=0.2, label='P10-P90 Interval')

    plt.title(f"Probabilistic {forecast_type.upper()} Forecast - {forecast_date} ({dataset_id})")
    plt.xlabel('Time')
    plt.ylabel(f'{forecast_type.capitalize()} (MW)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 8. 保存图表和结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"{results_dir}/prob_forecast_{forecast_date}_{timestamp}.png"
    csv_path = f"{results_dir}/prob_forecast_{forecast_date}_{timestamp}.csv"
    
    plt.savefig(plot_path, dpi=300)
    print(f"概率预测图表已保存至 {plot_path}")
    plt.close()
    
    results_df.to_csv(csv_path, index=False)
    print(f"概率预测结果已保存至 {csv_path}")

    return results_df

# Example usage (if run directly)
if __name__ == '__main__':
    try:
        perform_probabilistic_forecast(
            data_path='data/timeseries_load_上海.csv', # Example path
            forecast_date='2024-04-01',
            dataset_id='上海',
            forecast_type='load',
            quantiles=[0.1, 0.5, 0.9],
            interval_minutes=15
        )
    except FileNotFoundError:
        print("错误：测试需要示例数据文件 data/timeseries_load_上海.csv")
        print("并且需要已训练的概率模型 models/prob_convtrans/load/上海/")
    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc() 