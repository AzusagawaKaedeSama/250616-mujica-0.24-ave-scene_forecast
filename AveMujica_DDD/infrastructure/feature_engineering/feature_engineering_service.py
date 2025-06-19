import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List

from AveMujica_DDD.application.ports.i_feature_engineering_service import IFeatureEngineeringService

class DefaultFeatureEngineeringService(IFeatureEngineeringService):
    """
    一个通用的、由配置驱动的特征工程服务。
    """

    def preprocess_for_prediction(self, data: pd.DataFrame, target_column: str, feature_columns: List[str]) -> pd.DataFrame:
        """
        根据模型定义的 `target_column` 和 `feature_columns` 来进行特征工程。
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("输入DataFrame的索引必须是DatetimeIndex。")

        # 1. 只选择模型需要的原始特征列
        all_required_cols = list(set([target_column] + feature_columns))
        
        # 检查缺失的列并添加默认值
        missing_cols = [col for col in all_required_cols if col not in data.columns]
        if missing_cols:
            print(f"--- [FeatureEngineering] 发现缺失的特征列: {missing_cols}，将添加默认值 ---")
            for col in missing_cols:
                if col == 'pressure':
                    data[col] = 1013.25  # 标准大气压
                elif col == 'wind_direction':
                    data[col] = 0.0  # 北风方向
                elif col == 'solar_radiation':
                    data[col] = 0.0  # 无太阳辐射（夜间默认值）
                else:
                    data[col] = 0.0  # 其他缺失特征默认为0
        
        df = data[all_required_cols].copy()

        # 2. 生成基础时间特征 (这些总是被添加)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # 3. 只为负荷列添加关键的滞后特征（简化版本）
        if target_column in df.columns:
            # 只添加最重要的滞后特征
            df[f'{target_column}_lag_1'] = df[target_column].shift(1)     # 前15分钟
            df[f'{target_column}_lag_4'] = df[target_column].shift(4)     # 前1小时
            df[f'{target_column}_lag_96'] = df[target_column].shift(96)   # 前24小时
            df[f'{target_column}_lag_672'] = df[target_column].shift(672) # 前7天
            
            # 添加简单的滑动平均
            df[f'{target_column}_ma_4'] = df[target_column].rolling(window=4).mean()
            df[f'{target_column}_ma_24'] = df[target_column].rolling(window=24).mean()
            df[f'{target_column}_ma_96'] = df[target_column].rolling(window=96).mean()
            
            # 添加滑动标准差
            df[f'{target_column}_std_4'] = df[target_column].rolling(window=4).std()
            df[f'{target_column}_std_24'] = df[target_column].rolling(window=24).std()
        
        # 4. 为天气特征添加更多衍生特征
        weather_features = [col for col in feature_columns if col != target_column]
        for col_name in weather_features:
            if col_name in df.columns:
                # 添加滞后特征
                df[f'{col_name}_lag_1'] = df[col_name].shift(1)
                df[f'{col_name}_lag_4'] = df[col_name].shift(4)
                # 添加移动平均
                df[f'{col_name}_ma_4'] = df[col_name].rolling(window=4).mean()
                df[f'{col_name}_ma_24'] = df[col_name].rolling(window=24).mean()
                # 添加移动标准差
                df[f'{col_name}_std_4'] = df[col_name].rolling(window=4).std()
        
        # 5. 添加更多时间特征（精简版本以匹配训练时特征数量）
        # 删除以下特征以匹配训练时的特征数量（需要精确57个特征）
        # df['day_of_month'] = df.index.day

        # 6. 处理缺失值
        df = df.bfill().ffill() # 先用后值填充，再用前值填充
        df = df.fillna(0) # 对剩余的 (文件开头的) NaN 用0填充

        print(f"--- [FeatureEngineering] 特征工程完成，共生成 {len(df.columns)} 个特征 ---")
        print(f"--- [FeatureEngineering] 特征列表: {list(df.columns)} ---")
        return df

    def preprocess_for_training(self, data: pd.DataFrame, target_column: str, feature_columns: List[str]) -> tuple[pd.DataFrame, pd.Series]:
        """为训练准备特征和标签。"""
        if target_column not in data.columns:
            raise KeyError(f"训练数据中必须包含目标列 '{target_column}'。")
        
        processed_df = self.preprocess_for_prediction(data, target_column, feature_columns)
        
        y = processed_df[target_column]
        X = processed_df.drop(columns=[target_column])
        
        return X, y 