import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_weather_data(weather_file, interpolate=True, filter_outliers=True, add_engineered_features=True):
    """
    加载和预处理天气数据
    
    Args:
        weather_file: 天气数据文件路径
        interpolate: 是否插值缺失值
        filter_outliers: 是否过滤异常值
        add_engineered_features: 是否添加工程特征
        
    Returns:
        pandas.DataFrame: 处理后的天气数据
    """
    # 加载数据
    print(f"从 {weather_file} 加载天气数据...")
    df = pd.read_csv(weather_file)
    
    # 确保timestamp列是日期时间类型
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("天气数据缺少timestamp列")
        
    original_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [c for c in numeric_columns if c != 'timestamp']
    
    # 处理缺失值
    if interpolate:
        print(f"处理缺失值... 缺失值数量: {df[numeric_columns].isna().sum().sum()}")
        # 先按时间排序
        df = df.sort_values('timestamp')
        # 对数值列进行插值
        df[numeric_columns] = df[numeric_columns].interpolate(method='time')
        # 处理前后的缺失值
        df[numeric_columns] = df[numeric_columns].fillna(method='bfill').fillna(method='ffill')
        
    # 过滤异常值
    if filter_outliers:
        print("过滤异常值...")
        for col in numeric_columns:
            # 计算z得分
            z_scores = stats.zscore(df[col], nan_policy='omit')
            abs_z_scores = np.abs(z_scores)
            # 找出z得分大于3的数据点（潜在异常值）
            outliers = abs_z_scores > 3
            outlier_count = outliers.sum()
            if outlier_count > 0:
                print(f"  - {col}: 检测到 {outlier_count} 个异常值")
                # 使用均值替换异常值
                df.loc[outliers, col] = df[col].mean()
    
    # 添加工程特征
    if add_engineered_features:
        print("添加天气工程特征...")
        df = add_weather_engineered_features(df)
        
    # 打印数据摘要
    print(f"天气数据处理完成: {len(df)} 行, {len(df.columns)} 列")
    print(f"原始列: {original_columns}")
    print(f"添加的特征列: {[c for c in df.columns if c not in original_columns]}")
    
    return df

def add_weather_engineered_features(df):
    """
    添加天气特征工程
    
    Args:
        df: 包含基础天气数据的数据帧
        
    Returns:
        pandas.DataFrame: 带工程特征的数据帧
    """
    # 复制数据帧，避免修改原始数据
    df = df.copy()
    
    # 检查标准天气列是否存在
    columns = df.columns
    
    # 1. 计算热指数（如果有温度和湿度）
    if 'temperature' in columns and 'humidity' in columns:
        df['heat_index'] = calculate_heat_index(df['temperature'], df['humidity'])
        
    # 2. 计算体感温度（如果有温度和风速）
    if 'temperature' in columns and 'wind_speed' in columns:
        df['feels_like'] = calculate_feels_like(df['temperature'], df['wind_speed'])
    
    # 3. 计算露点温度（如果有温度和湿度）
    if 'temperature' in columns and 'humidity' in columns:
        df['dew_point'] = calculate_dew_point(df['temperature'], df['humidity'])
    
    # 4. 计算24小时温差（如果有timestamp和温度）
    if 'timestamp' in columns and 'temperature' in columns:
        # 确保时间戳已排序
        df = df.sort_values('timestamp')
        # 计算24小时前的温度
        df['temp_24h_ago'] = df['temperature'].shift(24)
        # 计算温差
        df['temp_change_24h'] = df['temperature'] - df['temp_24h_ago']
        # 删除临时列
        df = df.drop('temp_24h_ago', axis=1)
        
    # 5. 添加天气状况的二值特征（如果有weather_condition列）
    if 'weather_condition' in columns:
        # 常见天气状况
        weather_categories = ['clear', 'cloudy', 'rain', 'snow', 'fog', 'thunderstorm']
        
        # 低字符化以统一大小写
        df['weather_condition'] = df['weather_condition'].str.lower()
        
        # 为每种天气添加二值列
        for category in weather_categories:
            col_name = f'is_{category}'
            df[col_name] = df['weather_condition'].str.contains(category, na=False).astype(int)
    
    # 6. 计算温度趋势
    if 'timestamp' in columns and 'temperature' in columns:
        # 计算前后3小时的温度差（如果时间间隔为1小时）
        df['temp_trend_3h'] = (df['temperature'].shift(-3) - df['temperature'].shift(3)) / 6
        # 填充缺失值
        df['temp_trend_3h'] = df['temp_trend_3h'].fillna(0)
    
    return df

def calculate_heat_index(temperature, humidity):
    """
    计算热指数（体感温度）
    
    Args:
        temperature: 温度（摄氏度）
        humidity: 相对湿度（百分比）
        
    Returns:
        numpy.ndarray: 热指数（摄氏度）
    """
    # 转换为华氏温度进行计算
    t_f = temperature * 9/5 + 32
    rh = humidity
    
    # 热指数公式
    hi = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    
    # 对于高温高湿情况，使用更复杂的公式
    mask = (t_f >= 80) & (rh >= 40)
    
    # 完整热指数公式系数
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -0.00683783
    c6 = -0.05481717
    c7 = 0.00122874
    c8 = 0.00085282
    c9 = -0.00000199
    
    # 计算复杂公式
    hi_complex = c1 + c2*t_f + c3*rh + c4*t_f*rh + c5*t_f**2 + c6*rh**2 + c7*t_f**2*rh + c8*t_f*rh**2 + c9*t_f**2*rh**2
    
    # 根据条件使用简单或复杂公式
    hi = np.where(mask, hi_complex, hi)
    
    # 转回摄氏度
    hi_c = (hi - 32) * 5/9
    
    return hi_c

def calculate_feels_like(temperature, wind_speed):
    """
    计算体感温度（考虑风寒指数）
    
    Args:
        temperature: 温度（摄氏度）
        wind_speed: 风速（m/s）
        
    Returns:
        numpy.ndarray: 体感温度（摄氏度）
    """
    # 转换为华氏温度
    t_f = temperature * 9/5 + 32
    # 转换为英里/小时
    v = wind_speed * 2.237
    
    # 风寒指数公式（华氏度）
    wci = np.where(
        t_f <= 50, 
        35.74 + 0.6215*t_f - 35.75*(v**0.16) + 0.4275*t_f*(v**0.16),
        t_f
    )
    
    # 转回摄氏度
    wci_c = (wci - 32) * 5/9
    
    return wci_c

def calculate_dew_point(temperature, humidity):
    """
    计算露点温度
    
    Args:
        temperature: 温度（摄氏度）
        humidity: 相对湿度（百分比）
        
    Returns:
        numpy.ndarray: 露点温度（摄氏度）
    """
    # Magnus-Tetens近似公式
    alpha = 17.27
    beta = 237.7  # °C
    
    # 计算露点
    gamma = (alpha * temperature) / (beta + temperature) + np.log(humidity / 100.0)
    dew_point = (beta * gamma) / (alpha - gamma)
    
    return dew_point

def plot_weather_features(weather_df, features=None, start_date=None, end_date=None, figsize=(15, 12)):
    """
    绘制天气特征的时间序列
    
    Args:
        weather_df: 天气数据帧
        features: 要绘制的特征（如果为None，则选择所有数值特征）
        start_date: 开始日期
        end_date: 结束日期
        figsize: 图形大小
    """
    # 复制数据帧以避免修改
    df = weather_df.copy()
    
    # 确保时间戳列为日期时间类型
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("天气数据缺少timestamp列")
    
    # 过滤日期范围
    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]
    
    # 如果未指定特征，选择所有数值特征（除了timestamp）
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f != 'timestamp']
    
    # 限制特征数量以避免图表太复杂
    if len(features) > 12:
        print(f"警告: 太多特征 ({len(features)}). 仅显示前12个.")
        features = features[:12]
    
    # 设置绘图
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # 绘制每个特征
    for i, feature in enumerate(features):
        if i < len(axes):  # 确保有足够的轴
            ax = axes[i]
            ax.plot(df['timestamp'], df[feature])
            ax.set_title(f'{feature}')
            ax.set_xlabel('时间')
            ax.set_ylabel('值')
            ax.grid(True)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 隐藏未使用的子图
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def merge_weather_and_load(load_df, weather_df, interpolate_weather=True):
    """
    合并负荷和天气数据
    
    Args:
        load_df: 负荷数据帧，必须包含timestamp列
        weather_df: 天气数据帧，必须包含timestamp列
        interpolate_weather: 是否对天气数据进行线性插值以匹配所有负荷时间戳
        
    Returns:
        pandas.DataFrame: 合并后的数据帧
    """
    # 确保时间戳列为日期时间类型
    load_df = load_df.copy()
    weather_df = weather_df.copy()
    
    if 'timestamp' not in load_df.columns or 'timestamp' not in weather_df.columns:
        raise ValueError("负荷数据和天气数据都必须包含timestamp列")
    
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    print(f"负荷数据范围: {load_df['timestamp'].min()} 到 {load_df['timestamp'].max()}")
    print(f"天气数据范围: {weather_df['timestamp'].min()} 到 {weather_df['timestamp'].max()}")
    
    if interpolate_weather:
        # 确保天气数据按时间排序
        weather_df = weather_df.sort_values('timestamp')
        
        # 创建包含所有负荷时间戳的新天气数据帧
        new_weather = pd.DataFrame({'timestamp': load_df['timestamp'].unique()})
        
        # 对每个天气特征进行插值
        for col in weather_df.columns:
            if col != 'timestamp':
                # 创建插值函数
                interp_func = pd.Series(index=weather_df['timestamp'], data=weather_df[col].values)
                # 应用插值
                new_weather[col] = new_weather['timestamp'].map(lambda x: interp_func.asof(x))
        
        # 使用插值后的天气数据
        weather_df = new_weather
    
    # 合并数据
    merged_df = pd.merge(load_df, weather_df, on='timestamp', how='left')
    
    # 检查缺失值
    missing_pct = merged_df.isna().mean() * 100
    has_missing = missing_pct.max() > 0
    
    if has_missing:
        print("合并后数据的缺失值百分比:")
        for col in missing_pct[missing_pct > 0].index:
            print(f"  - {col}: {missing_pct[col]:.2f}%")
            
        # 填充缺失值
        print("使用前向填充和后向填充方法填充缺失值")
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    
    return merged_df

def create_lagged_weather_features(df, weather_features, lag_hours=[1, 3, 6, 24]):
    """
    创建滞后的天气特征
    
    Args:
        df: 包含天气特征的数据帧，必须按时间排序
        weather_features: 要创建滞后特征的天气特征列表
        lag_hours: 滞后小时数列表
        
    Returns:
        pandas.DataFrame: 带有滞后天气特征的数据帧
    """
    # 复制数据帧以避免修改原始数据
    df = df.copy()
    
    # 确保数据帧按时间排序
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # 为每个天气特征创建滞后版本
    for feature in weather_features:
        if feature in df.columns:
            for lag in lag_hours:
                lag_name = f"{feature}_lag{lag}"
                df[lag_name] = df[feature].shift(lag)
    
    # 丢弃包含NaN的行（由于滞后而创建）
    df_no_na = df.dropna()
    
    print(f"添加了滞后特征后，从 {len(df)} 行减少到 {len(df_no_na)} 行（由于滞后操作）")
    
    return df_no_na

def filter_weather_features(df, correlation_threshold=0.1, max_features=10, target_col='load'):
    """
    通过与目标的相关性筛选天气特征
    
    Args:
        df: 包含天气特征和目标变量的数据帧
        correlation_threshold: 最小相关性阈值
        max_features: 保留的最大特征数
        target_col: 目标列名
        
    Returns:
        list: 选择的特征列表
    """
    # 检查目标列是否存在
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在数据帧中")
    
    # 选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除目标列本身
    feature_cols = [c for c in numeric_cols if c != target_col]
    
    # 计算与目标的相关性
    correlations = df[feature_cols].corrwith(df[target_col]).abs()
    
    # 根据相关性对特征排序
    sorted_corrs = correlations.sort_values(ascending=False)
    
    # 过滤超过阈值的特征
    filtered_features = sorted_corrs[sorted_corrs > correlation_threshold]
    
    # 限制特征数量
    if len(filtered_features) > max_features:
        filtered_features = filtered_features[:max_features]
    
    selected_features = filtered_features.index.tolist()
    
    print(f"根据与 '{target_col}' 的相关性选择了 {len(selected_features)} 个特征:")
    for feature, corr in filtered_features.items():
        print(f"  - {feature}: {corr:.4f}")
    
    return selected_features 