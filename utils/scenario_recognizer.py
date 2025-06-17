import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ScenarioRecognizer:
    """基于预测数据识别电力系统运行场景"""
    
    def __init__(self, system_params=None):
        """
        初始化场景识别器
        
        参数:
        - system_params: 系统参数字典，包含容量阈值、爬坡率等
        """
        # 默认系统参数
        self.default_params = {
            'capacity': 10000,  # MW
            'max_ramp_rate': 300,  # MW/15min
            'renewable_penetration_threshold': 0.6,  # 60%
            'pv_threshold': 1000,  # MW
            'peak_hours': (8, 22),
            'valley_hours': (0, 7)
        }
        
        self.system_params = system_params or self.default_params
        
        # 定义场景规则
        self.scenario_rules = {
            'high_peak_load': self._high_peak_load_rule,
            'high_ramp_rate': self._high_ramp_rate_rule,
            'pv_sudden_drop': self._pv_sudden_drop_rule,
            'high_renewable_penetration': self._high_renewable_penetration_rule,
            'valley_load': self._valley_load_rule
        }
        
        # 场景描述
        self.scenario_descriptions = {
            'high_peak_load': '高峰负荷压力场景',
            'high_ramp_rate': '净负荷高爬坡率场景',
            'pv_sudden_drop': '光伏出力骤降场景',
            'high_renewable_penetration': '高可再生能源占比场景',
            'valley_load': '低谷负荷场景'
        }
    
    def _high_peak_load_rule(self, load_forecast, pv_forecast, wind_forecast):
        """高峰负荷场景识别规则"""
        # 使用P90负荷预测(高估计)
        if 'p90' in load_forecast.columns:
            peak_load = load_forecast['p90']
        else:
            peak_load = load_forecast['predicted']
            
        # 计算净负荷（如果有光伏和风电预测）
        net_load = peak_load.copy()
        if 'predicted' in pv_forecast.columns:
            net_load = net_load - pv_forecast['predicted']
        if 'predicted' in wind_forecast.columns:
            net_load = net_load - wind_forecast['predicted']
        
        # 计算当天净负荷均值，检查是否为空
        if len(net_load) == 0:
            return {'is_scenario': pd.Series([], dtype=bool)}
        
        net_load_mean = net_load.mean()
        
        # 检查均值是否为0或NaN，防止后续计算错误
        if net_load_mean == 0 or pd.isna(net_load_mean):
            net_load_mean = peak_load.mean() if peak_load.mean() > 0 else 1000  # 默认值
        
        # 设置相对于均值的高峰阈值：超过均值的140%（更严格一些）
        relative_threshold = net_load_mean * 1.40
        
        # 保留系统容量的绝对阈值作为上限保护
        absolute_threshold = self.system_params['capacity'] * 0.90
        
        # 实际使用的阈值是两者的最小值
        threshold = min(relative_threshold, absolute_threshold)
        
        # 添加时间限制，只在白天8点到21点之间考虑高峰负荷
        # 由于load_forecast现在已经有datetime作为索引，直接使用索引
        hours = load_forecast.index.hour
        is_peak_hour = (hours >= 8) & (hours <= 21)
        
        # 计算是否为高峰负荷
        is_scenario = (net_load > threshold) & is_peak_hour
        
        # 确保返回Series有与load_forecast相同的索引
        is_scenario = pd.Series(is_scenario, index=load_forecast.index)
        
        return {
            'is_scenario': is_scenario,
            'threshold': threshold,
            'value': net_load
        }
    
    def _high_ramp_rate_rule(self, load_forecast, pv_forecast, wind_forecast):
        """高爬坡率场景识别规则"""
        # 检查索引长度
        if len(load_forecast.index) == 0:
            return {'is_scenario': pd.Series([], dtype=bool)}
        
        # 计算净负荷 = 负荷 - 光伏 - 风电
        net_load = load_forecast['predicted'].copy()
        if 'predicted' in pv_forecast.columns:
            net_load = net_load - pv_forecast['predicted']
        if 'predicted' in wind_forecast.columns:
            net_load = net_load - wind_forecast['predicted']
        
        # 计算相邻时刻的净负荷差值（爬坡率）
        ramp_rates = net_load.diff().abs()
        
        # 排除第一个NaN值
        valid_ramp_rates = ramp_rates.dropna()
        
        # 数据检查
        if len(valid_ramp_rates) == 0:
            return {'is_scenario': pd.Series(False, index=load_forecast.index)}
        
        # 计算当天的平均爬坡率和最大爬坡率
        mean_ramp_rate = valid_ramp_rates.mean()
        
        # 设置相对爬坡率阈值：超过平均爬坡率的2.5倍（更严格）
        relative_threshold = mean_ramp_rate * 2.5
        
        # 避免过低阈值
        min_threshold = self.system_params['max_ramp_rate'] * 0.3  # 至少是最大爬坡率的30%
        relative_threshold = max(relative_threshold, min_threshold)
        
        # 同时，要求爬坡率至少占当前净负荷的一定比例（比如10%）才算显著
        significant_pct_threshold = 0.12  # 稍微提高显著性要求
        
        # 获取爬坡率
        ramp_rates = net_load.diff()
        
        # 根据当前净负荷计算显著性
        # 避免除以0，使用填充值
        net_load_filled = net_load.copy()
        net_load_filled = net_load_filled.replace(0, float('nan')).fillna(net_load_filled.mean())
        ramp_significance = ramp_rates.abs() / net_load_filled.abs()
        
        # 判断是否超过阈值
        is_high_ramp = ramp_rates.abs() > relative_threshold
        is_significant_pct = ramp_significance > significant_pct_threshold
        
        # 爬坡需要同时满足绝对爬坡率高和相对爬坡率显著
        is_scenario = is_high_ramp & is_significant_pct
        
        # 填充第一个时间点（NaN）为False
        if len(is_scenario) > 0:
            is_scenario.iloc[0] = False
        
        # 只考虑早晨的上升斜坡和晚上的下降斜坡作为特殊场景
        hours = load_forecast.index.hour
        # 上午爬坡有意义的时段 (6-10点)
        is_morning_ramp = (hours >= 6) & (hours <= 10) & (ramp_rates > 0)
        # 晚上爬坡有意义的时段 (17-21点)
        is_evening_ramp = (hours >= 17) & (hours <= 21) & (ramp_rates < 0)
        # 只在这些时段的爬坡才考虑为特殊场景
        is_scenario = is_scenario & (is_morning_ramp | is_evening_ramp)
        
        return {
            'is_scenario': is_scenario,
            'threshold': relative_threshold,
            'value': ramp_rates.abs()
        }
    
    def _pv_sudden_drop_rule(self, load_forecast, pv_forecast, wind_forecast):
        """光伏骤降场景识别规则"""
        # 检查是否有光伏预测数据
        if 'predicted' not in pv_forecast.columns or pv_forecast.empty:
            return {'is_scenario': pd.Series(False, index=load_forecast.index)}
        
        # 提取光伏预测值
        pv_values = pv_forecast['predicted']
        
        # 检查数据有效性
        if len(pv_values) == 0 or pv_values.max() == 0:
            return {'is_scenario': pd.Series(False, index=load_forecast.index)}
        
        # 计算当天光伏平均输出和最大输出
        pv_values_positive = pv_values[pv_values > 0]
        pv_mean = pv_values_positive.mean() if len(pv_values_positive) > 0 else 0
        pv_max = pv_values.max()
        
        # 只考虑白天时段的光伏变化(10点到16点)
        hours = pv_forecast.index.hour
        is_daytime = (hours >= 10) & (hours <= 16)
        
        # 计算光伏变化率
        pv_change_rate = pv_values.pct_change()
        
        # 设置变化率阈值：下降超过-35%才视为骤降（更严格一些）
        drop_threshold = -0.35
        
        # 光伏输出需要达到一定基准才计算骤降
        # 基准阈值为当天最大光伏输出的50%或平均值的150%，取较高者（更严格）
        base_threshold = max(pv_max * 0.50, pv_mean * 1.50)
        
        # 光伏占负荷比例需要达到一定水平才有影响
        load_values = load_forecast['predicted']
        pv_to_load_ratio = pv_values / load_values.replace(0, np.nan).fillna(load_values.mean())
        is_significant_pv = pv_to_load_ratio > 0.18  # 光伏至少占负荷18%（更严格）
        
        # 四个条件：1.白天时段 2.光伏有显著下降 3.光伏基准值足够高 4.光伏占比显著
        is_sudden_drop = (pv_change_rate < drop_threshold)
        is_high_base = pv_values.shift(1) > base_threshold
        
        # 组合所有条件
        is_scenario = is_daytime & is_sudden_drop & is_high_base & is_significant_pv
        
        # 处理第一个点的NaN
        if len(is_scenario) > 0:
            is_scenario.iloc[0] = False
        
        return {
            'is_scenario': is_scenario,
            'threshold': drop_threshold,
            'value': pv_change_rate
        }
    
    def _high_renewable_penetration_rule(self, load_forecast, pv_forecast, wind_forecast):
        """高可再生能源占比场景识别规则"""
        # 检查是否有负荷、光伏和风电数据
        if ('predicted' not in load_forecast.columns or 
            'predicted' not in pv_forecast.columns or 
            'predicted' not in wind_forecast.columns):
            return {'is_scenario': pd.Series(False, index=load_forecast.index)}
        
        # 检查数据有效性
        if len(load_forecast.index) == 0:
            return {'is_scenario': pd.Series([], dtype=bool)}
        
        # 提取预测值
        load_values = load_forecast['predicted']
        pv_values = pv_forecast['predicted'] if 'predicted' in pv_forecast.columns else 0
        wind_values = wind_forecast['predicted'] if 'predicted' in wind_forecast.columns else 0
        
        # 计算可再生能源总量和占比
        renewable_values = pv_values + wind_values
        
        # 避免除以0
        load_values_filled = load_values.replace(0, np.nan).fillna(load_values.mean())
        renewable_ratio = renewable_values / load_values_filled
        
        # 处理可能的无效值
        if renewable_ratio.isna().all() or renewable_ratio.max() == 0:
            return {'is_scenario': pd.Series(False, index=load_forecast.index)}
        
        # 计算当天可再生能源平均占比和最大占比
        mean_ratio = renewable_ratio.mean()
        
        # 设置高可再生能源占比阈值：超过当天平均占比的170%（更严格）
        ratio_threshold = mean_ratio * 1.70
        
        # 设置最小阈值，确保有实际意义
        min_ratio_threshold = 0.40  # 至少达到40%的可再生能源占比
        ratio_threshold = max(ratio_threshold, min_ratio_threshold)
        
        # 白天时段限制(上午9点到下午5点)
        hours = load_forecast.index.hour
        is_daytime = (hours >= 9) & (hours <= 17)
        
        # 要求可再生能源总量达到系统容量的30%以上
        capacity_threshold = self.system_params['capacity'] * 0.30
        is_significant_capacity = renewable_values > capacity_threshold
        
        # 要求负荷也有足够规模，不是在极低负荷时期
        load_threshold = self.system_params['capacity'] * 0.40
        is_significant_load = load_values > load_threshold
        
        # 判断是否为高可再生能源场景
        is_high_ratio = renewable_ratio > ratio_threshold
        
        # 综合条件：在白天时段 + 高可再生能源占比 + 可再生能源总量显著 + 负荷规模显著
        is_scenario = is_daytime & is_high_ratio & is_significant_capacity & is_significant_load
        
        return {
            'is_scenario': is_scenario,
            'threshold': ratio_threshold,
            'value': renewable_ratio
        }
    
    def _valley_load_rule(self, load_forecast, pv_forecast, wind_forecast):
        """低谷负荷场景识别规则"""
        # 检查数据有效性
        if len(load_forecast.index) == 0 or 'predicted' not in load_forecast.columns:
            return {'is_scenario': pd.Series([], dtype=bool)}
        
        # 使用预测负荷
        load_values = load_forecast['predicted']
        
        # 提取小时
        hours = load_forecast.index.hour
        
        # 限制只在深夜时段识别低谷负荷(凌晨1-5点)
        is_valley_hour = (hours >= 1) & (hours <= 5)
        
        # 计算当天平均负荷和最低负荷
        avg_load = load_values.mean()
        min_load = load_values.min()
        
        # 设置相对低谷阈值：低于平均负荷的60%
        relative_threshold = avg_load * 0.60
        
        # 要求负荷接近当天最低点(与最低点相差不超过15%)
        min_load_threshold = min_load * 1.15
        
        # 判断是否为低谷场景
        is_low_load = load_values < relative_threshold
        is_near_min = load_values < min_load_threshold
        
        # 同时满足三个条件：是低谷时段 + 低于相对阈值 + 接近当天最低点
        is_scenario = is_valley_hour & is_low_load & is_near_min
        
        return {
            'is_scenario': is_scenario,
            'threshold': relative_threshold,
            'value': load_values
        }
    
    def identify_scenarios(self, load_forecast, pv_forecast, wind_forecast):
        """
        识别电力系统运行场景
        
        参数:
        - load_forecast: 负荷预测DataFrame
        - pv_forecast: 光伏预测DataFrame
        - wind_forecast: 风电预测DataFrame
        
        返回:
        - 场景识别结果DataFrame
        """
        # 创建副本避免修改原始数据
        load_df = load_forecast.copy()
        pv_df = pv_forecast.copy()
        wind_df = wind_forecast.copy()
        
        # 确保所有输入具有datetime列
        if 'datetime' not in load_df.columns:
            raise ValueError("负荷预测数据必须包含'datetime'列")
        
        # 将datetime设为索引之前确保格式一致(转换为datetime对象)
        load_df['datetime'] = pd.to_datetime(load_df['datetime'])
        
        # 确保光伏和风电数据包含datetime列并格式一致
        if 'datetime' in pv_df.columns:
            pv_df['datetime'] = pd.to_datetime(pv_df['datetime'])
        else:
            # 如果没有datetime列，使用负荷的datetime
            pv_df['datetime'] = load_df['datetime'].values
        
        if 'datetime' in wind_df.columns:
            wind_df['datetime'] = pd.to_datetime(wind_df['datetime'])
        else:
            # 如果没有datetime列，使用负荷的datetime
            wind_df['datetime'] = load_df['datetime'].values
        
        # 现在设置datetime为索引
        load_df.set_index('datetime', inplace=True)
        pv_df.set_index('datetime', inplace=True)
        wind_df.set_index('datetime', inplace=True)
        
        # 确保三个数据集具有相同的索引
        common_index = load_df.index.intersection(pv_df.index).intersection(wind_df.index)
        load_df = load_df.loc[common_index]
        pv_df = pv_df.loc[common_index]
        wind_df = wind_df.loc[common_index]
        
        # 初始化场景DataFrame
        scenario_df = pd.DataFrame(index=common_index)
        
        # 应用场景规则
        for scenario_name, rule_func in self.scenario_rules.items():
            result = rule_func(load_df, pv_df, wind_df)
            scenario_df[scenario_name] = result['is_scenario']
        
        # 重置索引，使datetime成为列
        scenario_df.reset_index(inplace=True)
        
        return scenario_df
    
    def get_regulation_targets(self, scenario_df, load_forecast, pv_forecast, wind_forecast):
        """
        基于识别的场景生成调节目标
        
        参数:
        - scenario_df: 场景识别结果DataFrame
        - load_forecast: 负荷预测DataFrame
        - pv_forecast: 光伏预测DataFrame
        - wind_forecast: 风电预测DataFrame
        
        返回:
        - 调节目标DataFrame
        """
        # 创建副本避免修改原始数据
        scenario_df = scenario_df.copy()
        load_df = load_forecast.copy()
        pv_df = pv_forecast.copy()
        wind_df = wind_forecast.copy()
        
        # 确保scenario_df有datetime列
        if 'datetime' not in scenario_df.columns:
            raise ValueError("场景识别结果必须包含'datetime'列")
        
        # 确保所有数据帧有可比较的datetime格式
        scenario_df['datetime'] = pd.to_datetime(scenario_df['datetime'])
        
        # 创建调节目标DataFrame
        regulation_df = pd.DataFrame()
        regulation_df['datetime'] = scenario_df['datetime']
        
        # 将负荷、光伏、风电数据合并到regulation_df
        # 确保datetime格式一致，然后基于datetime合并
        # 如果输入已经使用datetime为索引，首先重置索引
        if isinstance(load_df.index, pd.DatetimeIndex):
            load_df = load_df.reset_index()
        if isinstance(pv_df.index, pd.DatetimeIndex):
            pv_df = pv_df.reset_index()
        if isinstance(wind_df.index, pd.DatetimeIndex):
            wind_df = wind_df.reset_index()
        
        # 确保datetime列存在并格式一致
        if 'datetime' in load_df.columns:
            load_df['datetime'] = pd.to_datetime(load_df['datetime'])
            load_df_merged = load_df[['datetime', 'predicted']]
            regulation_df = pd.merge(regulation_df, load_df_merged, 
                                   on='datetime', how='left', suffixes=('', '_load'))
            regulation_df.rename(columns={'predicted': 'load'}, inplace=True)
        else:
            regulation_df['load'] = 0
        
        if 'datetime' in pv_df.columns and 'predicted' in pv_df.columns:
            pv_df['datetime'] = pd.to_datetime(pv_df['datetime'])
            pv_df_merged = pv_df[['datetime', 'predicted']]
            regulation_df = pd.merge(regulation_df, pv_df_merged, 
                                   on='datetime', how='left', suffixes=('', '_pv'))
            regulation_df.rename(columns={'predicted': 'pv'}, inplace=True)
        else:
            regulation_df['pv'] = 0
        
        if 'datetime' in wind_df.columns and 'predicted' in wind_df.columns:
            wind_df['datetime'] = pd.to_datetime(wind_df['datetime'])
            wind_df_merged = wind_df[['datetime', 'predicted']]
            regulation_df = pd.merge(regulation_df, wind_df_merged, 
                                   on='datetime', how='left', suffixes=('', '_wind'))
            regulation_df.rename(columns={'predicted': 'wind'}, inplace=True)
        else:
            regulation_df['wind'] = 0
        
        # 计算净负荷
        regulation_df['net_load'] = regulation_df['load'] - regulation_df['pv'] - regulation_df['wind']
        
        # 确保净负荷不为负数
        regulation_df['net_load'] = regulation_df['net_load'].clip(lower=0)
        
        # 识别的场景类型
        regulation_df['scenario_type'] = '正常运行'  # 默认场景
        regulation_df['regulation_need'] = '维持系统平衡'  # 默认调节需求
        
        # 合并场景识别结果到regulation_df
        # 将场景掩码视为布尔列，避免NaN干扰判断
        scenario_df = scenario_df.fillna(False)
        
        # 处理高峰负荷场景
        mask_high_peak = scenario_df['high_peak_load'].astype(bool)
        regulation_df.loc[mask_high_peak, 'scenario_type'] = '高峰负荷压力'
        regulation_df.loc[mask_high_peak, 'regulation_need'] = '增加可用容量，需求侧响应'
        
        # 低谷负荷场景
        mask_valley = scenario_df['valley_load'].astype(bool)
        regulation_df.loc[mask_valley, 'scenario_type'] = '低谷负荷'
        regulation_df.loc[mask_valley, 'regulation_need'] = '减少基础发电，提高负荷水平'
        
        # 高爬坡率场景
        mask_ramp = scenario_df['high_ramp_rate'].astype(bool)
        regulation_df.loc[mask_ramp, 'scenario_type'] = '净负荷爬坡'
        regulation_df.loc[mask_ramp, 'regulation_need'] = '快速响应资源辅助'
        
        # 光伏骤降场景
        mask_pv_drop = scenario_df['pv_sudden_drop'].astype(bool)
        regulation_df.loc[mask_pv_drop, 'scenario_type'] = '光伏波段风险'
        regulation_df.loc[mask_pv_drop, 'regulation_need'] = '预留向上备用，确能补偿'
        
        # 高可再生能源渗透率场景
        mask_renewable = scenario_df['high_renewable_penetration'].astype(bool)
        regulation_df.loc[mask_renewable, 'scenario_type'] = '高可再生比例'
        regulation_df.loc[mask_renewable, 'regulation_need'] = '合成惯量服务，保障系统稳定性'
        
        # 初始化调节目标列
        regulation_df['upward_reserve'] = 0.0  # 向上备用需求
        regulation_df['downward_reserve'] = 0.0  # 向下备用需求
        regulation_df['demand_response'] = 0.0  # 需求响应需求
        regulation_df['energy_storage'] = 0.0  # 储能需求
        
        # 计算向上备用 (基于光伏波段风险或高爬坡)
        regulation_df.loc[mask_pv_drop, 'upward_reserve'] = regulation_df.loc[mask_pv_drop, 'pv'] * 0.3  # 光伏的30%
        
        # 为高爬坡场景设置向上备用
        if any(mask_ramp):
            # 使用净负荷变化来设置爬坡场景的备用
            net_load_diff = regulation_df['net_load'].diff().abs().fillna(0)
            regulation_df.loc[mask_ramp, 'upward_reserve'] = regulation_df.loc[mask_ramp, 'upward_reserve'].combine(
                net_load_diff.loc[mask_ramp] * 0.5,  # 净负荷变化的50%
                lambda x, y: max(x, y)
            )
        
        # 计算向下备用 (基于高可再生或低谷)
        if any(mask_renewable):
            renewable_sum = regulation_df.loc[mask_renewable, 'pv'] + regulation_df.loc[mask_renewable, 'wind']
            regulation_df.loc[mask_renewable, 'downward_reserve'] = renewable_sum * 0.15  # 可再生能源的15%
        
        # 需求响应 (基于高峰负荷)
        regulation_df.loc[mask_high_peak, 'demand_response'] = regulation_df.loc[mask_high_peak, 'load'] * 0.05  # 负荷的5%
        
        # 储能需求 (基于净负荷爬坡和光伏骤降)
        combined_mask = mask_ramp | mask_pv_drop
        regulation_df.loc[combined_mask, 'energy_storage'] = regulation_df.loc[combined_mask, 'upward_reserve'] * 0.7  # 向上备用的70%
        
        # 为正常场景添加基本调节需求
        mask_normal = ~(mask_high_peak | mask_valley | mask_ramp | mask_pv_drop | mask_renewable)
        regulation_df.loc[mask_normal, 'upward_reserve'] = regulation_df.loc[mask_normal, 'load'] * 0.03  # 负荷的3%
        regulation_df.loc[mask_normal, 'downward_reserve'] = regulation_df.loc[mask_normal, 'load'] * 0.02  # 负荷的2%
        
        return regulation_df