import uuid
from datetime import date
import traceback

print("开始导入基础设施层...")

# 基础设施层 - 真实组件的导入
from AveMujica_DDD.infrastructure.adapters.pytorch_prediction_engine import PyTorchPredictionEngine
print("PyTorchPredictionEngine 导入成功")

from AveMujica_DDD.infrastructure.data_providers.csv_provider import CsvDataProvider
print("CsvDataProvider 导入成功")

from AveMujica_DDD.infrastructure.repositories.in_memory_repository import (
    InMemoryForecastRepository,
    InMemoryWeatherScenarioRepository,
)
print("InMemory repositories 导入成功")

from AveMujica_DDD.infrastructure.repositories.sqlite_repository import SQLiteModelRepository
print("SQLiteModelRepository 导入成功")

print("开始导入应用层和领域层...")

# 应用层和领域层的导入
from AveMujica_DDD.application.services.forecast_service import ForecastService
print("ForecastService 导入成功")

from AveMujica_DDD.domain.services.uncertainty_calculation_service import UncertaintyCalculationService
print("UncertaintyCalculationService 导入成功")

from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
print("PredictionModel 导入成功")

print("所有导入完成！")

def main():
    """
    应用的组装线和主入口点 (Composition Root)。
    在这里，我们创建所有具体类的实例，并将它们连接在一起。
    """
    print("--- 正在组装应用 ---")

    # 1. 实例化基础设施层的真实组件
    db_path = "data/mujica_prod.db"
    model_repo = SQLiteModelRepository(db_path=db_path)
    forecast_repo = InMemoryForecastRepository()
    weather_scenario_repo = InMemoryWeatherScenarioRepository()
    data_provider = CsvDataProvider() 
    prediction_engine = PyTorchPredictionEngine()

    # 在第一次运行时，向数据库中添加一个默认模型
    print("检查默认模型...")
    if not model_repo.find_by_name_and_version("default_load_model", "1.0"):
        print("--- 数据库中无默认模型，正在添加... ---")
        
        # 定义模型的自我描述
        target_column = 'load'
        feature_columns = [
            'load',             # 注意：目标列本身也可以作为输入特征（用于计算滞后项等）
            'temperature',      # 原来是 weather_temperature_c
            'humidity',         # 原来是 weather_relative_humidity
            'pressure',         # 缺失，将添加默认值
            'wind_speed',       # 原来是 weather_wind_speed
            'wind_direction',   # 缺失，将添加默认值
            'precipitation',    # 原来是 weather_precipitation_mm
            'solar_radiation'   # 缺失，将添加默认值
        ]

        default_model = PredictionModel(
            name="default_load_model",
            version="1.0",
            forecast_type=ForecastType.LOAD,
            file_path="models/convtrans_weather/load/上海/best_model.pth",
            target_column=target_column,
            feature_columns=feature_columns,
            description="一个默认的、用于演示的负荷预测模型"
        )
        model_repo.save(default_model)
        print("默认模型已保存")
    else:
        print("默认模型已存在")

    # 2. 实例化领域服务
    uncertainty_service = UncertaintyCalculationService()

    # 3. 实例化应用服务，并将所有依赖项注入
    forecast_service = ForecastService(
        forecast_repo=forecast_repo,
        model_repo=model_repo,
        weather_scenario_repo=weather_scenario_repo,
        weather_provider=data_provider,
        prediction_engine=prediction_engine,
        uncertainty_service=uncertainty_service,
    )
    print("--- 应用组装完成 ---\n")


    # 4. 运行一个示例用例
    print("--- 正在执行日前负荷预测用例 ---")
    try:
        test_model = model_repo.find_by_name_and_version("default_load_model", "1.0")
        if not test_model:
            print("错误：找不到预加载的测试模型。")
            return
            
        result_dto = forecast_service.create_day_ahead_load_forecast(
            province="上海",
            target_date=date(2024, 1, 10), # 选择一个有足够历史数据的日期
            model_id=test_model.model_id,
            historical_days=8 # 指定需要8天的历史数据来计算特征
        )
        print("\n--- 预测成功！ ---")
        print(f"预测ID: {result_dto.forecast_id}")
        print(f"省份: {result_dto.province}")
        print(f"使用模型: {result_dto.model_name}")
        print(f"天气场景: {result_dto.scenario_type}")
        print(f"第一个数据点: {result_dto.time_series[0] if result_dto.time_series else 'N/A'}")
        print("--------------------")

    except Exception as e:
        print(f"\n--- 执行用例时发生错误 ---")
        print(f"错误信息: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 要运行此文件，请在项目根目录下执行: python main.py
    print("开始执行main函数...")
    main()
    print("main函数执行完成") 