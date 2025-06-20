from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

class ScenarioType(str, Enum):
    """
    17种天气场景枚举，基于实际电力系统运行数据和气象特征定义。
    """
    # 极端天气场景 (4种)
    EXTREME_STORM_RAIN = "极端暴雨"
    EXTREME_HOT_HUMID = "极端高温高湿"
    EXTREME_STRONG_WIND = "极端大风"
    EXTREME_HEAVY_RAIN = "特大暴雨"
    
    # 典型场景 (3种，基于真实数据分析)
    TYPICAL_GENERAL_NORMAL = "一般正常场景"
    TYPICAL_RAINY_LOW_LOAD = "多雨低负荷"
    TYPICAL_MILD_HUMID_HIGH_LOAD = "温和高湿高负荷"
    
    # 普通天气变种 (4种)
    NORMAL_SPRING_MILD = "春季温和"
    NORMAL_SUMMER_COMFORTABLE = "夏季舒适"
    NORMAL_AUTUMN_STABLE = "秋季平稳"
    NORMAL_WINTER_MILD = "冬季温和"
    
    # 原有基础场景 (6种，保持兼容)
    EXTREME_HOT = "极端高温"
    EXTREME_COLD = "极端寒冷"
    HIGH_WIND_SUNNY = "大风晴朗"
    CALM_CLOUDY = "无风阴天"
    MODERATE_NORMAL = "温和正常"
    STORM_RAIN = "暴雨大风"

@dataclass
class WeatherScenario:
    """
    天气场景聚合根。
    代表一种已识别的天气模式及其对电力系统的影响规则。
    包含场景识别规则、不确定性调整规则和运行建议。
    """
    scenario_type: ScenarioType
    description: str
    uncertainty_multiplier: float
    typical_features: Dict[str, float]  # 典型气象特征
    power_system_impact: str  # 对电力系统的影响
    operation_suggestions: str  # 运行建议
    historical_frequency: float = 0.0  # 历史出现频率(%)
    seasonal_pattern: str = ""  # 季节性模式

    def is_extreme_weather(self) -> bool:
        """
        业务规则：判断当前场景是否为极端天气。
        """
        return self.scenario_type.name.startswith("EXTREME")

    def is_high_uncertainty(self) -> bool:
        """
        业务规则：判断当前场景是否为高不确定性场景。
        """
        return self.uncertainty_multiplier > 2.0
    
    def get_backup_capacity_recommendation(self) -> float:
        """
        业务规则：根据场景特点推荐备用容量比例。
        """
        if self.is_extreme_weather():
            return 0.35  # 极端天气需要35%备用容量
        elif self.uncertainty_multiplier > 1.5:
            return 0.25  # 高不确定性场景需要25%备用容量
        else:
            return 0.15  # 正常场景15%备用容量

# 完整的17种天气场景预定义实例
PREDEFINED_SCENARIOS = {
    # 极端天气场景 (4种)
    ScenarioType.EXTREME_STORM_RAIN: WeatherScenario(
        scenario_type=ScenarioType.EXTREME_STORM_RAIN,
        description="极端暴雨天气，降水>30mm，湿度>90%，系统运行极不稳定",
        uncertainty_multiplier=3.5,
        typical_features={"temperature": 25.0, "humidity": 95.0, "wind_speed": 6.0, "precipitation": 35.0},
        power_system_impact="系统运行极不稳定，负荷模式异常",
        operation_suggestions="启动应急预案，增加35%备用容量，加强设备巡检",
        historical_frequency=2.1,
        seasonal_pattern="夏季多发，6-8月"
    ),
    
    ScenarioType.EXTREME_HOT_HUMID: WeatherScenario(
        scenario_type=ScenarioType.EXTREME_HOT_HUMID,
        description="极端高温高湿天气，温度>32°C，湿度>80%，空调负荷极高",
        uncertainty_multiplier=3.0,
        typical_features={"temperature": 35.0, "humidity": 85.0, "wind_speed": 3.0, "precipitation": 0.0},
        power_system_impact="空调负荷极高，电网压力巨大，可能出现负荷激增",
        operation_suggestions="全力运行发电机组，准备需求响应措施，监控线路温度",
        historical_frequency=3.8,
        seasonal_pattern="夏季高温期，7-8月"
    ),
    
    ScenarioType.EXTREME_STRONG_WIND: WeatherScenario(
        scenario_type=ScenarioType.EXTREME_STRONG_WIND,
        description="极端大风天气，风速>10m/s，风电出力波动极大",
        uncertainty_multiplier=2.8,
        typical_features={"temperature": 20.0, "humidity": 60.0, "wind_speed": 12.0, "precipitation": 5.0},
        power_system_impact="风电出力波动极大，系统调节压力增加",
        operation_suggestions="加强风电功率预测，准备快速调节资源，监控电网稳定性",
        historical_frequency=4.2,
        seasonal_pattern="春秋季多发，3-5月、9-11月"
    ),
    
    ScenarioType.EXTREME_HEAVY_RAIN: WeatherScenario(
        scenario_type=ScenarioType.EXTREME_HEAVY_RAIN,
        description="特大暴雨天气，降水>50mm，湿度>95%，负荷模式异常",
        uncertainty_multiplier=4.0,
        typical_features={"temperature": 22.0, "humidity": 98.0, "wind_speed": 4.0, "precipitation": 60.0},
        power_system_impact="负荷模式异常，预测困难，设备故障风险高",
        operation_suggestions="启动最高级别应急预案，做好防汛准备，备用电源就位",
        historical_frequency=1.5,
        seasonal_pattern="梅雨季节，6-7月"
    ),
    
    # 典型场景 (3种，基于真实数据分析)
    ScenarioType.TYPICAL_GENERAL_NORMAL: WeatherScenario(
        scenario_type=ScenarioType.TYPICAL_GENERAL_NORMAL,
        description="一般正常场景，天气条件平稳，对应历史数据中的一般场景0",
        uncertainty_multiplier=1.0,
        typical_features={"temperature": 22.0, "humidity": 65.0, "wind_speed": 4.0, "precipitation": 0.0},
        power_system_impact="系统运行平稳，负荷预测相对准确",
        operation_suggestions="正常运行模式，常规备用容量即可",
        historical_frequency=33.3,
        seasonal_pattern="四季均有，春秋季较多"
    ),
    
    ScenarioType.TYPICAL_RAINY_LOW_LOAD: WeatherScenario(
        scenario_type=ScenarioType.TYPICAL_RAINY_LOW_LOAD,
        description="多雨低负荷场景，对应历史数据中占比43.4%的多雨低负荷模式",
        uncertainty_multiplier=1.3,
        typical_features={"temperature": 18.0, "humidity": 85.0, "wind_speed": 3.0, "precipitation": 15.0},
        power_system_impact="负荷水平相对较低，但天气因素增加不确定性",
        operation_suggestions="适当降低发电出力，注意负荷波动，保持灵活调节能力",
        historical_frequency=43.4,
        seasonal_pattern="秋冬季多发，10-12月"
    ),
    
    ScenarioType.TYPICAL_MILD_HUMID_HIGH_LOAD: WeatherScenario(
        scenario_type=ScenarioType.TYPICAL_MILD_HUMID_HIGH_LOAD,
        description="温和高湿高负荷场景，对应历史数据中占比23.2%的温和高湿高负荷模式",
        uncertainty_multiplier=1.4,
        typical_features={"temperature": 28.0, "humidity": 78.0, "wind_speed": 2.5, "precipitation": 2.0},
        power_system_impact="负荷水平较高，湿度影响舒适度需求",
        operation_suggestions="增加20%备用容量，关注用电高峰时段，优化经济调度",
        historical_frequency=23.2,
        seasonal_pattern="夏季多发，5-9月"
    ),
    
    # 普通天气变种 (4种)
    ScenarioType.NORMAL_SPRING_MILD: WeatherScenario(
        scenario_type=ScenarioType.NORMAL_SPRING_MILD,
        description="春季温和天气，15-22°C，负荷平稳，适合设备检修",
        uncertainty_multiplier=0.9,
        typical_features={"temperature": 18.0, "humidity": 60.0, "wind_speed": 4.0, "precipitation": 5.0},
        power_system_impact="负荷平稳，系统运行稳定，不确定性较低",
        operation_suggestions="利用负荷平稳期安排设备检修，优化运行方式",
        historical_frequency=12.5,
        seasonal_pattern="春季，3-5月"
    ),
    
    ScenarioType.NORMAL_SUMMER_COMFORTABLE: WeatherScenario(
        scenario_type=ScenarioType.NORMAL_SUMMER_COMFORTABLE,
        description="夏季舒适天气，25-30°C，空调负荷适中",
        uncertainty_multiplier=1.1,
        typical_features={"temperature": 27.0, "humidity": 70.0, "wind_speed": 3.5, "precipitation": 0.0},
        power_system_impact="空调负荷适中，整体需求平稳上升",
        operation_suggestions="正常夏季运行模式，关注峰谷差变化",
        historical_frequency=15.8,
        seasonal_pattern="夏季，6-8月"
    ),
    
    ScenarioType.NORMAL_AUTUMN_STABLE: WeatherScenario(
        scenario_type=ScenarioType.NORMAL_AUTUMN_STABLE,
        description="秋季平稳天气，18-25°C，系统最稳定期",
        uncertainty_multiplier=0.8,
        typical_features={"temperature": 21.0, "humidity": 55.0, "wind_speed": 3.0, "precipitation": 2.0},
        power_system_impact="系统最稳定期，预测精度最高",
        operation_suggestions="最佳运行期，可进行系统优化试验",
        historical_frequency=18.2,
        seasonal_pattern="秋季，9-11月"
    ),
    
    ScenarioType.NORMAL_WINTER_MILD: WeatherScenario(
        scenario_type=ScenarioType.NORMAL_WINTER_MILD,
        description="冬季温和天气，8-15°C，采暖负荷适中",
        uncertainty_multiplier=1.2,
        typical_features={"temperature": 12.0, "humidity": 50.0, "wind_speed": 5.0, "precipitation": 1.0},
        power_system_impact="采暖负荷适中，整体需求稳定",
        operation_suggestions="注意采暖负荷变化，保持供热电厂稳定运行",
        historical_frequency=14.3,
        seasonal_pattern="冬季，12-2月"
    ),
    
    # 原有基础场景 (6种，保持兼容)
    ScenarioType.EXTREME_HOT: WeatherScenario(
        scenario_type=ScenarioType.EXTREME_HOT,
        description="极端高温天气，温度>35°C，高空调负荷",
        uncertainty_multiplier=2.5,
        typical_features={"temperature": 37.0, "humidity": 60.0, "wind_speed": 2.0, "precipitation": 0.0},
        power_system_impact="负荷剧烈波动，高不确定性，可能出现尖峰负荷",
        operation_suggestions="全网备用容量就位，启动需求响应，加强负荷监控",
        historical_frequency=5.2,
        seasonal_pattern="盛夏，7-8月"
    ),
    
    ScenarioType.EXTREME_COLD: WeatherScenario(
        scenario_type=ScenarioType.EXTREME_COLD,
        description="极端寒冷天气，温度<0°C，高采暖负荷",
        uncertainty_multiplier=2.0,
        typical_features={"temperature": -3.0, "humidity": 40.0, "wind_speed": 6.0, "precipitation": 0.0},
        power_system_impact="负荷增加，中高不确定性，供暖负荷大幅上升",
        operation_suggestions="确保燃料供应，加强供热机组运行，预防设备结冰",
        historical_frequency=3.8,
        seasonal_pattern="严冬，12-1月"
    ),
    
    ScenarioType.HIGH_WIND_SUNNY: WeatherScenario(
        scenario_type=ScenarioType.HIGH_WIND_SUNNY,
        description="大风晴朗天气，风速>8m/s，高太阳辐射",
        uncertainty_multiplier=0.8,
        typical_features={"temperature": 20.0, "humidity": 45.0, "wind_speed": 10.0, "precipitation": 0.0},
        power_system_impact="新能源大发，低不确定性，系统负荷压力小",
        operation_suggestions="充分利用新能源出力，调整火电机组出力，优化系统经济性",
        historical_frequency=8.7,
        seasonal_pattern="春秋季，4-5月、9-10月"
    ),
    
    ScenarioType.CALM_CLOUDY: WeatherScenario(
        scenario_type=ScenarioType.CALM_CLOUDY,
        description="无风阴天天气，风速<3m/s，低太阳辐射",
        uncertainty_multiplier=1.5,
        typical_features={"temperature": 18.0, "humidity": 75.0, "wind_speed": 2.0, "precipitation": 0.0},
        power_system_impact="新能源出力低，中等不确定性，需要更多常规机组",
        operation_suggestions="增加常规机组出力，减少对新能源的依赖，保持系统平衡",
        historical_frequency=11.4,
        seasonal_pattern="冬春季，11-3月"
    ),
    
    ScenarioType.MODERATE_NORMAL: WeatherScenario(
        scenario_type=ScenarioType.MODERATE_NORMAL,
        description="温和正常天气条件，系统平稳运行",
        uncertainty_multiplier=1.0,
        typical_features={"temperature": 20.0, "humidity": 60.0, "wind_speed": 4.0, "precipitation": 0.0},
        power_system_impact="系统平稳运行，基准不确定性",
        operation_suggestions="标准运行模式，常规备用容量配置",
        historical_frequency=25.6,
        seasonal_pattern="四季均有"
    ),
    
    ScenarioType.STORM_RAIN: WeatherScenario(
        scenario_type=ScenarioType.STORM_RAIN,
        description="暴雨大风天气，强降水，恶劣天气",
        uncertainty_multiplier=3.0,
        typical_features={"temperature": 24.0, "humidity": 90.0, "wind_speed": 8.0, "precipitation": 25.0},
        power_system_impact="系统风险高，最高不确定性，设备故障概率增加",
        operation_suggestions="启动恶劣天气应急预案，增加30%备用容量，加强安全监控",
        historical_frequency=6.3,
        seasonal_pattern="夏季暴雨季，6-8月"
    )
}

def get_scenario_by_type(scenario_type: ScenarioType) -> Optional[WeatherScenario]:
    """
    根据场景类型获取预定义的天气场景实例。
    """
    return PREDEFINED_SCENARIOS.get(scenario_type)

def get_all_scenarios() -> Dict[ScenarioType, WeatherScenario]:
    """
    获取所有预定义的天气场景。
    """
    return PREDEFINED_SCENARIOS.copy()

def get_extreme_scenarios() -> Dict[ScenarioType, WeatherScenario]:
    """
    获取所有极端天气场景。
    """
    return {k: v for k, v in PREDEFINED_SCENARIOS.items() if v.is_extreme_weather()}

def get_scenarios_by_season(season: str) -> Dict[ScenarioType, WeatherScenario]:
    """
    根据季节获取相关的天气场景。
    
    Args:
        season: 季节名称 ("春季", "夏季", "秋季", "冬季")
    """
    season_keywords = {
        "春季": ["春季", "3-5月", "4-5月"],
        "夏季": ["夏季", "6-8月", "7-8月", "5-9月"],
        "秋季": ["秋季", "9-11月", "10-12月"],
        "冬季": ["冬季", "12-2月", "11-3月", "12-1月"]
    }
    
    keywords = season_keywords.get(season, [])
    if not keywords:
        return {}
    
    result = {}
    for scenario_type, scenario in PREDEFINED_SCENARIOS.items():
        for keyword in keywords:
            if keyword in scenario.seasonal_pattern:
                result[scenario_type] = scenario
                break
    
    return result 