from dataclasses import dataclass
from enum import Enum

class ScenarioType(str, Enum):
    """
    使用字符串枚举，使其与JSON序列化更兼容。
    """
    EXTREME_STORM_RAIN = "极端暴雨"
    EXTREME_HOT_HUMID = "极端高温高湿"
    EXTREME_STRONG_WIND = "极端大风"
    EXTREME_HEAVY_RAIN = "特大暴雨"
    TYPICAL_GENERAL_NORMAL = "一般正常场景"
    TYPICAL_RAINY_LOW_LOAD = "多雨低负荷"
    TYPICAL_MILD_HUMID_HIGH_LOAD = "温和高湿高负荷"
    NORMAL_SPRING_MILD = "春季温和"
    NORMAL_SUMMER_COMFORTABLE = "夏季舒适"
    NORMAL_AUTUMN_STABLE = "秋季平稳"
    NORMAL_WINTER_MILD = "冬季温和"
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
    """
    scenario_type: ScenarioType
    description: str
    uncertainty_multiplier: float

    def is_extreme_weather(self) -> bool:
        """
        一个业务规则，用于判断当前场景是否为极端天气。
        """
        return self.scenario_type.name.startswith("EXTREME")

# 我们可以预定义一些场景实例，这些可以从配置文件或数据库加载。
# 这只是一个例子，实际应用中会通过仓储(Repository)来获取。
PREDEFINED_SCENARIOS = {
    ScenarioType.EXTREME_HOT_HUMID: WeatherScenario(
        scenario_type=ScenarioType.EXTREME_HOT_HUMID,
        description="温度>32°C，湿度>80%，空调负荷极高，电网压力巨大",
        uncertainty_multiplier=3.0
    ),
    ScenarioType.MODERATE_NORMAL: WeatherScenario(
        scenario_type=ScenarioType.MODERATE_NORMAL,
        description="温和天气条件，系统平稳运行",
        uncertainty_multiplier=1.0
    ),
} 