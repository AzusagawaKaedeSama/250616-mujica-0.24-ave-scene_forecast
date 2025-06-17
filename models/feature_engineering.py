# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """综合特征工程处理器"""
    
    def __init__(self, 
                 outliers_ratio=0.05,
                 svr_kernel='rbf',
                 elastic_net_alpha=0.1,
                 encoding_columns=None,
                 n_bins=10):
        self.outliers_ratio = outliers_ratio
        self.svr_kernel = svr_kernel
        self.elastic_net_alpha = elastic_net_alpha
        self.encoding_columns = encoding_columns or []
        self.n_bins = n_bins
        
        self.pipeline = Pipeline([
            ('outlier_detection', OutlierHandler(contamination=outliers_ratio)),
            ('missing_value_imputation', SVRImputer(kernel=svr_kernel)),
            ('feature_selection', ElasticNetSelector(alpha=elastic_net_alpha)),
            ('categorical_encoding', ColumnTransformer(
                transformers=[
                    ('interval', IntervalEncoder(n_bins=n_bins), encoding_columns),
                    ('label', LabelEncoder(), [col for col in encoding_columns if col not in self.interval_cols])
                ]
            )),
            ('scaler', StandardScaler())
        ])
    
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X, y=None):
        return self.pipeline.transform(X)

class OutlierHandler(BaseEstimator, TransformerMixin):
    """基于孤立森林的异常值处理器"""
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.model = None
    
    def fit(self, X, y=None):
        self.model = IsolationForest(contamination=self.contamination)
        self.model.fit(X)
        return self
    
    def transform(self, X, y=None):
        preds = self.model.predict(X)
        return X[preds == 1]

class SVRImputer(BaseEstimator, TransformerMixin):
    """基于LS-SVR的缺失值插补器"""
    def __init__(self, kernel='rbf'):
        self.kernel = kernel
        self.model = None
    
    def fit(self, X, y):
        # 创建监督学习数据对
        X_train = X[:-1]
        X_test = X[1:]
        y_train = X_train.copy()
        y_test = X_test.copy()
        
        self.model = SVR(kernel=self.kernel)
        self.model.fit(y_train, X_test)
        return self
    
    def transform(self, X):
        return self.model.predict(X)

class ElasticNetSelector(BaseEstimator, TransformerMixin):
    """基于弹性网络的稀疏特征选择器"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.selector = None
    
    def fit(self, X, y):
        self.selector = SelectFromModel(
            ElasticNetCV(alpha=self.alpha, cv=5),
            prefit=True
        )
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector.transform(X)

class IntervalEncoder(BaseEstimator, TransformerMixin):
    """基于分箱的区间编码器"""
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.encoders = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            unique_vals = sorted(X[col].unique())
            bins = np.linspace(unique_vals[0], unique_vals[-1], self.n_bins + 1)
            labels = np.digitize(X[col], bins) - 1
            self.encoders[col] = (bins, labels)
        return self
    
    def transform(self, X):
        transformed = X.copy()
        for col in X.columns:
            if col not in self.encoders:
                continue
            bins, labels = self.encoders[col]
            transformed[col] = labels
        return transformed