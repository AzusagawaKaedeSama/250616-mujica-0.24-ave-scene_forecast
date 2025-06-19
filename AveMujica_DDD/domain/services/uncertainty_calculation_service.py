from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario

class UncertaintyCalculationService:
    """
    领域服务：不确定性计算。
    
    这个服务封装了"天气场景感知的动态不确定性"计算逻辑。
    这是一个核心的业务规则，因为它不属于任何单个聚合，所以放在领域服务中。
    该服务是无状态的。
    """
    
    def calculate_bounds(
        self, 
        predicted_value: float, 
        base_uncertainty_rate: float, 
        scenario: WeatherScenario,
        time_period_adjustment: float = 1.0
    ) -> tuple[float, float]:
        """
        根据给定的预测值、基础不确定性比率和天气场景，计算动态的预测上限和下限。
        
        最终不确定性 = 基础不确定性 * 场景不确定性倍数 * 时段调整系数
        
        :param predicted_value: 模型的点预测值
        :param base_uncertainty_rate: 该预测类型的基础不确定性比率 (例如: 负荷为0.05)
        :param scenario: 匹配到的天气场景
        :param time_period_adjustment: 针对特殊时段（如高峰）的调整系数
        :return: (下限, 上限)
        """
        if not (0 <= base_uncertainty_rate <= 1):
            raise ValueError("基础不确定性比率必须在0和1之间")

        final_uncertainty_rate = (
            base_uncertainty_rate 
            * scenario.uncertainty_multiplier 
            * time_period_adjustment
        )
        
        uncertainty_value = predicted_value * final_uncertainty_rate
        
        lower_bound = predicted_value - uncertainty_value
        upper_bound = predicted_value + uncertainty_value
        
        return lower_bound, upper_bound 