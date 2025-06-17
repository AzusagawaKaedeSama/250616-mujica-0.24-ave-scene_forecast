import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class WeatherScenarioClassifier:
    """天气场景分类器，用于识别不同的天气场景并分配不确定性调整系数"""
    
    def __init__(self):
        """初始化天气场景分类器"""
        # 预定义场景及其不确定性调整系数
        self.scenarios = {
            'extreme_hot': {
                'name': '极端高温',
                'description': '温度异常高，可能导致空调负荷激增，系统压力大',
                'uncertainty_multiplier': 2.5,
                'risk_level': 'high',
                'typical_features': {
                    'temperature': '>35°C',
                    'humidity': '较高',
                    'wind_speed': '弱到中等',
                    'precipitation': '少或无',
                    'radiation': '强'
                }
            },
            'extreme_cold': {
                'name': '极端寒冷',
                'description': '温度异常低，可能导致采暖负荷激增，系统压力大',
                'uncertainty_multiplier': 2.0,
                'risk_level': 'high',
                'typical_features': {
                    'temperature': '<-5°C',
                    'humidity': '较低',
                    'wind_speed': '中等到强',
                    'precipitation': '可能有雪',
                    'radiation': '弱'
                }
            },
            'high_wind_sunny': {
                'name': '高风晴朗',
                'description': '风速较大且天气晴朗，风电出力高，光伏出力稳定',
                'uncertainty_multiplier': 0.8,
                'risk_level': 'low',
                'typical_features': {
                    'temperature': '适中',
                    'humidity': '较低',
                    'wind_speed': '>6m/s',
                    'precipitation': '无',
                    'radiation': '强'
                }
            },
            'calm_cloudy': {
                'name': '阴天微风',
                'description': '风速低且多云，风电光伏出力低且波动',
                'uncertainty_multiplier': 1.5,
                'risk_level': 'medium',
                'typical_features': {
                    'temperature': '适中',
                    'humidity': '中等',
                    'wind_speed': '<3m/s',
                    'precipitation': '无或小',
                    'radiation': '弱到中等'
                }
            },
            'moderate_normal': {
                'name': '温和正常',
                'description': '天气条件温和，系统运行平稳',
                'uncertainty_multiplier': 1.0,
                'risk_level': 'low',
                'typical_features': {
                    'temperature': '15-25°C',
                    'humidity': '适中',
                    'wind_speed': '3-5m/s',
                    'precipitation': '无',
                    'radiation': '中等'
                }
            },
            'storm_rain': {
                'name': '暴雨雷电',
                'description': '强降雨伴有雷电，系统运行不稳定，预测难度大',
                'uncertainty_multiplier': 3.0,
                'risk_level': 'high',
                'typical_features': {
                    'temperature': '变化大',
                    'humidity': '高',
                    'wind_speed': '变化大',
                    'precipitation': '>25mm/h',
                    'radiation': '弱'
                }
            }
        }
        
        # 基础不确定性水平（根据不同预测类型）
        self.base_uncertainties = {
            'load': 0.05,  # 负荷预测基础不确定性 5%
            'pv': 0.15,    # 光伏预测基础不确定性 15%
            'wind': 0.20   # 风电预测基础不确定性 20%
        }
        
        # 时间段调整系数
        self.time_period_adjustments = {
            'morning': 1.2,    # 早晨 (6:00-9:00)
            'midday': 0.9,     # 中午 (11:00-14:00)
            'evening_peak': 1.3, # 晚高峰 (18:00-21:00)
            'night': 1.1,      # 夜间 (22:00-5:00)
            'normal': 1.0      # 其他时段
        }
        
        self.scenario_history = []  # 历史场景记录
        
    def identify_scenario(self, weather_data):
        """
        根据天气数据识别当前天气场景
        
        参数:
        weather_data: 包含温度、湿度、风速等特征的DataFrame或字典
        
        返回:
        scenario_info: 场景信息字典，包含场景名称、描述、不确定性系数等
        """
        if isinstance(weather_data, pd.DataFrame):
            # 如果是DataFrame，提取均值作为特征
            features = {
                'temperature': weather_data.get('temperature', weather_data.get('temp', pd.Series([20]))).mean(),
                'humidity': weather_data.get('humidity', weather_data.get('relative_humidity', pd.Series([50]))).mean(),
                'wind_speed': weather_data.get('wind_speed', pd.Series([3])).mean(),
                'precipitation': weather_data.get('precipitation', weather_data.get('precip', pd.Series([0]))).mean(),
                'radiation': weather_data.get('radiation', weather_data.get('solar_radiation', pd.Series([500]))).mean()
            }
        else:
            # 如果是字典，直接使用
            features = weather_data
            
        # 为每个场景计算匹配分数
        scenario_scores = {}
        for scenario_id, scenario in self.scenarios.items():
            score = self._calculate_scenario_match_score(features, scenario_id)
            scenario_scores[scenario_id] = score
            
        # 找出得分最高的场景
        best_scenario_id = max(scenario_scores, key=scenario_scores.get)
        best_scenario = self.scenarios[best_scenario_id]
        
        # 构建详细的场景信息
        scenario_info = {
            'id': best_scenario_id,
            'name': best_scenario['name'],
            'description': best_scenario['description'],
            'uncertainty_multiplier': best_scenario['uncertainty_multiplier'],
            'risk_level': best_scenario['risk_level'],
            'match_score': scenario_scores[best_scenario_id],
            'weather_features': {
                'temperature': f"{features['temperature']:.1f}°C",
                'humidity': f"{features['humidity']:.1f}%",
                'wind_speed': f"{features['wind_speed']:.1f}m/s",
                'precipitation': f"{features['precipitation']:.1f}mm/h",
                'radiation': f"{features['radiation']:.1f}W/m²"
            },
            'typical_features': best_scenario['typical_features'],
            'all_scenario_scores': scenario_scores
        }
        
        return scenario_info
    
    def _calculate_scenario_match_score(self, features, scenario_id):
        """计算天气特征与特定场景的匹配分数"""
        # 不同特征的权重
        weights = {
            'temperature': 0.3,
            'humidity': 0.2,
            'wind_speed': 0.2,
            'radiation': 0.2,
            'precipitation': 0.1
        }
        
        score = 0
        
        # 极端高温场景
        if scenario_id == 'extreme_hot':
            temp_score = min(1.0, max(0, (features['temperature'] - 30) / 10))
            humidity_score = min(1.0, features['humidity'] / 80)
            wind_score = max(0, 1 - features['wind_speed'] / 10)
            precip_score = max(0, 1 - features['precipitation'] / 5)
            radiation_score = min(1.0, features['radiation'] / 800)
            
            score = (temp_score * weights['temperature'] + 
                    humidity_score * weights['humidity'] + 
                    wind_score * weights['wind_speed'] + 
                    precip_score * weights['precipitation'] + 
                    radiation_score * weights['radiation'])
        
        # 极端寒冷场景
        elif scenario_id == 'extreme_cold':
            temp_score = min(1.0, max(0, (5 - features['temperature']) / 15))
            humidity_score = max(0, 1 - features['humidity'] / 100)
            wind_score = min(1.0, features['wind_speed'] / 8)
            precip_score = 0.5  # 降雪不一定与温度直接相关
            radiation_score = max(0, 1 - features['radiation'] / 500)
            
            score = (temp_score * weights['temperature'] + 
                    humidity_score * weights['humidity'] + 
                    wind_score * weights['wind_speed'] + 
                    precip_score * weights['precipitation'] + 
                    radiation_score * weights['radiation'])
        
        # 高风晴朗场景
        elif scenario_id == 'high_wind_sunny':
            temp_score = max(0, 1 - abs(features['temperature'] - 20) / 15)
            humidity_score = max(0, 1 - features['humidity'] / 80)
            wind_score = min(1.0, max(0, (features['wind_speed'] - 5) / 5))
            precip_score = max(0, 1 - features['precipitation'] / 2)
            radiation_score = min(1.0, features['radiation'] / 700)
            
            score = (temp_score * weights['temperature'] + 
                    humidity_score * weights['humidity'] + 
                    wind_score * weights['wind_speed'] + 
                    precip_score * weights['precipitation'] + 
                    radiation_score * weights['radiation'])
        
        # 阴天微风场景
        elif scenario_id == 'calm_cloudy':
            temp_score = max(0, 1 - abs(features['temperature'] - 18) / 15)
            humidity_score = min(1.0, features['humidity'] / 70)
            wind_score = max(0, 1 - features['wind_speed'] / 5)
            precip_score = min(1.0, max(0, 1 - features['precipitation'] / 2))
            radiation_score = max(0, 1 - features['radiation'] / 500)
            
            score = (temp_score * weights['temperature'] + 
                    humidity_score * weights['humidity'] + 
                    wind_score * weights['wind_speed'] + 
                    precip_score * weights['precipitation'] + 
                    radiation_score * weights['radiation'])
        
        # 温和正常场景
        elif scenario_id == 'moderate_normal':
            temp_score = max(0, 1 - abs(features['temperature'] - 20) / 10)
            humidity_score = max(0, 1 - abs(features['humidity'] - 60) / 40)
            wind_score = max(0, 1 - abs(features['wind_speed'] - 4) / 4)
            precip_score = max(0, 1 - features['precipitation'] / 1)
            radiation_score = max(0, 1 - abs(features['radiation'] - 500) / 300)
            
            score = (temp_score * weights['temperature'] + 
                    humidity_score * weights['humidity'] + 
                    wind_score * weights['wind_speed'] + 
                    precip_score * weights['precipitation'] + 
                    radiation_score * weights['radiation'])
        
        # 暴雨雷电场景
        elif scenario_id == 'storm_rain':
            temp_score = 0.5  # 温度不是主要因素
            humidity_score = min(1.0, features['humidity'] / 90)
            wind_score = min(1.0, features['wind_speed'] / 10)
            precip_score = min(1.0, features['precipitation'] / 20)
            radiation_score = max(0, 1 - features['radiation'] / 300)
            
            score = (temp_score * weights['temperature'] + 
                    humidity_score * weights['humidity'] + 
                    wind_score * weights['wind_speed'] + 
                    precip_score * weights['precipitation'] + 
                    radiation_score * weights['radiation'])
        
        return score
    
    def get_uncertainty_parameters(self, scenario_info, forecast_type, datetime_obj=None):
        """
        根据场景信息、预测类型和时间获取不确定性参数
        
        参数:
        scenario_info: 场景信息字典
        forecast_type: 预测类型，'load', 'pv' 或 'wind'
        datetime_obj: 预测时间点，用于时间段调整
        
        返回:
        uncertainty_params: 不确定性参数字典
        """
        # 获取基础不确定性
        base_uncertainty = self.base_uncertainties.get(forecast_type, 0.05)
        
        # 获取场景调整系数
        scenario_multiplier = scenario_info['uncertainty_multiplier']
        
        # 获取时间段调整系数
        time_period_adjustment = 1.0
        if datetime_obj:
            hour = datetime_obj.hour
            if 6 <= hour < 9:
                time_period_adjustment = self.time_period_adjustments['morning']
            elif 11 <= hour < 14:
                time_period_adjustment = self.time_period_adjustments['midday']
            elif 18 <= hour < 21:
                time_period_adjustment = self.time_period_adjustments['evening_peak']
            elif hour >= 22 or hour < 5:
                time_period_adjustment = self.time_period_adjustments['night']
            else:
                time_period_adjustment = self.time_period_adjustments['normal']
        
        # 计算最终不确定性
        final_uncertainty = base_uncertainty * scenario_multiplier * time_period_adjustment
        
        # 构建不确定性参数
        uncertainty_params = {
            'base_uncertainty': base_uncertainty,
            'scenario_multiplier': scenario_multiplier,
            'time_period_adjustment': time_period_adjustment,
            'final_uncertainty': final_uncertainty,
            'forecast_type': forecast_type,
            'scenario': scenario_info['name'],
            'risk_level': scenario_info['risk_level'],
            'confidence_level': 0.9,  # 默认置信水平
            'explanation': {
                'base': f"基础{forecast_type}预测不确定性为{base_uncertainty*100:.1f}%",
                'scenario_effect': f"'{scenario_info['name']}'场景将不确定性调整为{scenario_multiplier}倍",
                'time_effect': f"当前时间段调整系数为{time_period_adjustment}",
                'final': f"最终不确定性水平为{final_uncertainty*100:.1f}%"
            }
        }
        
        return uncertainty_params
    
    def generate_explanation(self, scenario_info, uncertainty_params, forecast_results=None):
        """
        生成场景和不确定性的详细解释
        
        参数:
        scenario_info: 场景信息字典
        uncertainty_params: 不确定性参数字典
        forecast_results: 预测结果（可选）
        
        返回:
        explanation: 详细解释字典
        """
        forecast_type = uncertainty_params['forecast_type']
        forecast_type_names = {
            'load': '负荷',
            'pv': '光伏',
            'wind': '风电',
            'net_load': '净负荷'
        }
        forecast_type_name = forecast_type_names.get(forecast_type, forecast_type)
        
        # 场景影响分析
        scenario_impact = f"主导天气场景为\"{scenario_info['name']}\"：{scenario_info['description']}。该场景的不确定性倍数为{scenario_info['uncertainty_multiplier']}，"
        
        if scenario_info['uncertainty_multiplier'] > 1:
            scenario_impact += "增加了预测不确定性。"
        elif scenario_info['uncertainty_multiplier'] < 1:
            scenario_impact += "降低了预测不确定性。"
        else:
            scenario_impact += "对预测不确定性影响不大。"
            
        # 不确定性来源分析
        uncertainty_sources = [
            f"基础{forecast_type_name}预测不确定性：{uncertainty_params['base_uncertainty']*100:.1f}%",
            f"天气场景\"{scenario_info['name']}\"调整系数：{uncertainty_params['scenario_multiplier']}",
            f"时间段调整系数：{uncertainty_params['time_period_adjustment']}"
        ]
        
        # 计算方法说明
        calculation_method = f"最终不确定性 = 基础不确定性 × 场景调整系数 × 时间段调整系数 = {uncertainty_params['base_uncertainty']*100:.1f}% × {uncertainty_params['scenario_multiplier']} × {uncertainty_params['time_period_adjustment']} = {uncertainty_params['final_uncertainty']*100:.1f}%"
        
        # 运行建议
        operation_suggestions = []
        risk_level = scenario_info['risk_level']
        
        if risk_level == 'high':
            operation_suggestions.append(f"当前为高风险天气场景，{forecast_type_name}预测不确定性较大，建议增加备用容量")
            operation_suggestions.append(f"密切监控实时{forecast_type_name}变化，做好应急预案")
            operation_suggestions.append(f"考虑启用更多调峰资源应对可能的大幅波动")
        elif risk_level == 'medium':
            operation_suggestions.append(f"当前为中等风险天气场景，{forecast_type_name}预测波动中等")
            operation_suggestions.append(f"保持常规备用容量，定期检查{forecast_type_name}预测偏差")
        else:  # low
            operation_suggestions.append(f"当前为低风险天气场景，{forecast_type_name}预测相对稳定")
            operation_suggestions.append(f"可以采用常规运行策略，无需特别措施")
            
        # 天气特征分析
        weather_analysis = f"当前天气特征：温度{scenario_info['weather_features']['temperature']}，湿度{scenario_info['weather_features']['humidity']}，"
        weather_analysis += f"风速{scenario_info['weather_features']['wind_speed']}，降水{scenario_info['weather_features']['precipitation']}，"
        weather_analysis += f"辐射{scenario_info['weather_features']['radiation']}。"
        
        # 构建完整解释
        explanation = {
            'scenario_impact': scenario_impact,
            'weather_analysis': weather_analysis,
            'uncertainty_sources': uncertainty_sources,
            'calculation_method': calculation_method,
            'operation_suggestions': operation_suggestions,
            'risk_level': risk_level,
            'summary': f"基于当前\"{scenario_info['name']}\"天气场景，{forecast_type_name}预测的不确定性水平为{uncertainty_params['final_uncertainty']*100:.1f}%，置信水平为{uncertainty_params['confidence_level']*100:.0f}%。"
        }
        
        return explanation

    def analyze_forecast_period_scenarios(self, weather_forecast_df):
        """
        分析预测期间的天气场景
        
        参数:
        weather_forecast_df: 包含预测期间天气数据的DataFrame
        
        返回:
        scenario_analysis: 场景分析结果字典
        """
        scenario_analysis = {
            'scenarios': [],
            'dominant_scenario': None,
            'scenario_distribution': {},
            'uncertainty_profile': [],
            'risk_assessment': 'low',
            'recommendations': []
        }
        
        if weather_forecast_df.empty:
            logger.warning("天气数据为空，使用默认场景分析")
            # 返回默认的温和正常场景
            default_scenario_info = self.identify_scenario({
                'temperature': 20.0,
                'humidity': 60.0,
                'wind_speed': 3.0,
                'precipitation': 0.0,
                'radiation': 500.0
            })
            scenario_analysis['scenarios'] = [default_scenario_info]
            scenario_analysis['dominant_scenario'] = self.scenarios['moderate_normal']
            scenario_analysis['scenario_distribution'] = {
                'moderate_normal': {'count': 1, 'percentage': 100.0, 'scenario_name': '温和正常'}
            }
            scenario_analysis['uncertainty_profile'] = [1.0]
            return scenario_analysis
        
        scenarios_count = {}
        total_uncertainty = 0.0
        
        # 分析每个时间点的场景
        for idx, row in weather_forecast_df.iterrows():
            # 提取天气特征
            weather_data = {
                'temperature': row.get('temperature', 20.0),
                'humidity': row.get('humidity', 60.0),
                'wind_speed': row.get('wind_speed', 3.0),
                'precipitation': row.get('precipitation', 0.0),
                'radiation': row.get('radiation', row.get('solar_radiation', 500.0))
            }
            
            # 识别场景
            scenario_info = self.identify_scenario(weather_data)
            scenario_analysis['scenarios'].append(scenario_info)
            
            # 统计场景分布
            scenario_id = scenario_info['id']
            if scenario_id not in scenarios_count:
                scenarios_count[scenario_id] = 0
            scenarios_count[scenario_id] += 1
            
            # 累计不确定性
            uncertainty_multiplier = scenario_info['uncertainty_multiplier']
            total_uncertainty += uncertainty_multiplier
            scenario_analysis['uncertainty_profile'].append(uncertainty_multiplier)
        
        # 确定主导场景
        if scenarios_count:
            dominant_scenario_id = max(scenarios_count, key=scenarios_count.get)
            scenario_analysis['dominant_scenario'] = self.scenarios[dominant_scenario_id]
            
            # 计算场景分布百分比
            total_points = len(weather_forecast_df)
            for scenario_id, count in scenarios_count.items():
                scenario_analysis['scenario_distribution'][scenario_id] = {
                    'count': count,
                    'percentage': (count / total_points) * 100,
                    'scenario_name': self.scenarios[scenario_id]['name']
                }
        
        # 风险评估
        avg_uncertainty = total_uncertainty / len(weather_forecast_df) if len(weather_forecast_df) > 0 else 1.0
        if avg_uncertainty > 2.0:
            scenario_analysis['risk_assessment'] = 'high'
            scenario_analysis['recommendations'].append('建议增加备用容量，密切监控系统状态')
        elif avg_uncertainty > 1.5:
            scenario_analysis['risk_assessment'] = 'medium'
            scenario_analysis['recommendations'].append('建议适度调整调度策略，关注关键时段')
        else:
            scenario_analysis['risk_assessment'] = 'low'
            scenario_analysis['recommendations'].append('系统运行相对稳定，保持常规监控')
        
        logger.info(f"场景分析完成：主导场景 {scenario_analysis['dominant_scenario']['name'] if scenario_analysis['dominant_scenario'] else 'unknown'}，风险等级 {scenario_analysis['risk_assessment']}")
        
        return scenario_analysis

def create_weather_scenario_classifier():
    """创建天气场景分类器实例"""
    return WeatherScenarioClassifier() 