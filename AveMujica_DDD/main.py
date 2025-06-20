#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUJICA 天气感知负荷预测系统 - DDD架构版本
主入口文件：展示完整的预测流程
"""

import uuid
from datetime import date, datetime, timedelta

from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository
from AveMujica_DDD.domain.services.uncertainty_calculation_service import UncertaintyCalculationService

from AveMujica_DDD.application.services.forecast_service import ForecastService
from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider
from AveMujica_DDD.application.ports.i_prediction_engine import IPredictionEngine

from AveMujica_DDD.infrastructure.repositories.file_system_repos import (
    FileForecastRepository,
    FileModelRepository,
    FileWeatherScenarioRepository
)
from AveMujica_DDD.infrastructure.data_providers.real_weather_provider import RealWeatherProvider
from AveMujica_DDD.infrastructure.adapters.real_prediction_engine import RealPredictionEngine


class MujicaDDDSystem:
    """
    MUJICA 预测系统的主类，采用DDD架构设计。
    这个类负责系统的初始化和高级用例编排。
    """
    
    def __init__(self, use_real_implementations: bool = True):
        """
        初始化系统。
        
        Args:
            use_real_implementations: 是否使用真实实现（文件系统、真实模型）
        """
        print("🚀 初始化 MUJICA DDD 天气感知负荷预测系统...")
        
        self.use_real = use_real_implementations
        self._initialize_dependencies()
        
        print("✅ 系统初始化完成")

    def _initialize_dependencies(self):
        """初始化所有依赖项（依赖注入）"""
        
        # === 基础设施层 ===
        if self.use_real:
            self.forecast_repo: IForecastRepository = FileForecastRepository()
            self.model_repo: IModelRepository = FileModelRepository() 
            self.weather_scenario_repo: IWeatherScenarioRepository = FileWeatherScenarioRepository()
            self.weather_provider: IWeatherDataProvider = RealWeatherProvider()
            self.prediction_engine: IPredictionEngine = RealPredictionEngine()
        else:
            # 可以在这里使用内存实现进行测试
            from AveMujica_DDD.infrastructure.repositories.in_memory_repos import (
                InMemoryForecastRepository,
                InMemoryModelRepository,
                InMemoryWeatherScenarioRepository
            )
            self.forecast_repo = InMemoryForecastRepository()
            self.model_repo = InMemoryModelRepository()
            self.weather_scenario_repo = InMemoryWeatherScenarioRepository()
            self.weather_provider = RealWeatherProvider()  # 天气数据仍使用真实实现
            self.prediction_engine = RealPredictionEngine()

        # === 领域层 ===
        self.uncertainty_service = UncertaintyCalculationService()

        # === 应用层 ===
        self.forecast_service = ForecastService(
            forecast_repo=self.forecast_repo,
            model_repo=self.model_repo,
            weather_scenario_repo=self.weather_scenario_repo,
            weather_provider=self.weather_provider,
            prediction_engine=self.prediction_engine,
            uncertainty_service=self.uncertainty_service,
        )

    def run_example_forecast(self, province: str = "上海", days_ahead: int = 1):
        """
        运行一个完整的预测示例。
        
        Args:
            province: 省份名称
            days_ahead: 预测天数
        """
        print(f"\n📊 开始为 {province} 进行 {days_ahead} 天的负荷预测...")
        
        try:
            # 1. 设置预测参数
            start_date = date.today() + timedelta(days=1)  # 明天
            end_date = start_date + timedelta(days=days_ahead - 1)
            
            print(f"预测日期范围: {start_date} 到 {end_date}")
            
            # 2. 检查或创建模型
            model_id = self._ensure_model_exists(province)
            
            # 3. 执行预测
            forecast_result = self.forecast_service.create_day_ahead_load_forecast(
                province=province,
                start_date=start_date,
                end_date=end_date,
                model_id=model_id,
                historical_days=7
            )
            
            # 4. 显示结果
            self._display_forecast_result(forecast_result)
            
            return forecast_result
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _ensure_model_exists(self, province: str) -> uuid.UUID:
        """确保指定省份的负荷预测模型存在"""
        from AveMujica_DDD.domain.aggregates.prediction_model import ForecastType, PredictionModel
        
        # 查找现有模型
        existing_models = self.model_repo.find_by_type_and_region(
            forecast_type=ForecastType.LOAD,
            region=province
        )
        
        if existing_models:
            model = existing_models[0]
            print(f"🔍 找到现有模型: {model.name}")
            return model.model_id
        else:
            # 创建新的模型实例（在生产环境中应该是已训练的模型）
            model_id = uuid.uuid4()
            model_name = f"{province}_load_model"
            
            # 创建完整的PredictionModel对象并保存
            prediction_model = PredictionModel(
                model_id=model_id,
                name=model_name,
                version="1.0.0",
                forecast_type=ForecastType.LOAD,
                file_path=f"dummy/{model_name}.pth",
                target_column="load",
                feature_columns=["temperature", "humidity"],
                description=f"虚拟{province}负荷预测模型"
            )
            
            # 保存到仓储
            self.model_repo.save(prediction_model)
            print(f"🆕 创建新模型实例: {model_name}")
            
            return model_id

    def _display_forecast_result(self, forecast_dto):
        """展示预测结果"""
        print("\n" + "="*50)
        print("🎯 预测结果")
        print("="*50)
        
        print(f"预测ID: {forecast_dto.forecast_id}")
        print(f"省份: {forecast_dto.province}")
        print(f"创建时间: {forecast_dto.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"使用模型: {forecast_dto.model_name}")
        print(f"天气场景: {forecast_dto.scenario_type}")
        
        if forecast_dto.time_series:
            print(f"\n📈 预测数据点: {len(forecast_dto.time_series)} 个")
            
            # 显示前几个预测点作为示例
            sample_points = forecast_dto.time_series[:5]
            print("\n时间               | 预测值(MW)  | 下界(MW)    | 上界(MW)")
            print("-" * 60)
            
            for point in sample_points:
                time_str = point.timestamp.strftime('%Y-%m-%d %H:%M')
                pred_str = f"{point.value:.1f}".rjust(10)
                lower_str = f"{point.lower_bound:.1f}".rjust(10) if point.lower_bound else "N/A".rjust(10)
                upper_str = f"{point.upper_bound:.1f}".rjust(10) if point.upper_bound else "N/A".rjust(10)
                print(f"{time_str} | {pred_str} | {lower_str} | {upper_str}")
            
            if len(forecast_dto.time_series) > 5:
                print("...")
                
            # 计算统计信息
            values = [p.value for p in forecast_dto.time_series]
            avg_forecast = sum(values) / len(values)
            max_forecast = max(values)
            min_forecast = min(values)
            
            print(f"\n📊 统计信息:")
            print(f"   平均预测值: {avg_forecast:.1f} MW")
            print(f"   最大预测值: {max_forecast:.1f} MW")
            print(f"   最小预测值: {min_forecast:.1f} MW")
            
            # 如果有置信区间，计算平均区间宽度
            if forecast_dto.time_series[0].upper_bound is not None:
                interval_widths = [
                    p.upper_bound - p.lower_bound 
                    for p in forecast_dto.time_series 
                    if p.upper_bound is not None and p.lower_bound is not None
                ]
                if interval_widths:
                    avg_interval_width = sum(interval_widths) / len(interval_widths)
                    print(f"   平均置信区间宽度: {avg_interval_width:.1f} MW")
        
        print("="*50)

    def list_available_models(self):
        """列出所有可用的模型"""
        print("\n🔧 可用模型列表:")
        models = self.model_repo.list_all()
        
        if not models:
            print("   暂无可用模型")
        else:
            for model in models:
                print(f"   - {model.name} ({model.forecast_type.value})")

    def list_weather_scenarios(self):
        """列出所有天气场景"""
        print("\n🌤️  天气场景列表:")
        scenarios = self.weather_scenario_repo.list_all()
        
        for scenario in scenarios:
            print(f"   - {scenario.scenario_type.value}: {scenario.description}")
            print(f"     不确定性倍数: {scenario.uncertainty_multiplier}x")

    def start_api_server(self, port: int = 5001):
        """启动API服务器"""
        print(f"\n🌐 启动API服务器 (端口: {port})...")
        
        from AveMujica_DDD.api import DIContainer, create_app
        
        # 使用相同的依赖配置创建API服务器
        di_container = DIContainer(use_real_implementations=self.use_real)
        app = create_app(di_container)
        
        print(f"✅ API服务器已启动: http://localhost:{port}")
        print("   可用端点:")
        print("   - POST /api/predict - 执行预测")
        print("   - GET  /api/health  - 健康检查")
        print("   - GET  /api/models  - 模型列表")
        print("   - GET  /api/scenarios - 场景列表")
        
        app.run(debug=True, port=port, host='0.0.0.0')


def main():
    """主函数：演示完整的系统功能"""
    print("🌟 MUJICA 天气感知负荷预测系统 - DDD架构版本")
    print("=" * 60)
    
    # 初始化系统
    system = MujicaDDDSystem(use_real_implementations=True)
    
    # 显示系统状态
    system.list_available_models()
    system.list_weather_scenarios()
    
    # 运行预测示例
    forecast_result = system.run_example_forecast(province="上海", days_ahead=1)
    
    if forecast_result:
        print("\n✅ 预测任务完成！")
        print(f"   结果已保存到文件系统")
        print(f"   预测ID: {forecast_result.forecast_id}")
    
    # 询问是否启动API服务器
    print("\n" + "="*60)
    user_input = input("是否启动API服务器？(y/N): ").lower().strip()
    
    if user_input in ['y', 'yes', '是']:
        system.start_api_server()
    else:
        print("👋 感谢使用 MUJICA 预测系统！")


if __name__ == "__main__":
    main() 