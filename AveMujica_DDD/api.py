import uuid
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Domain Imports (reverted to absolute) ---
from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository
from AveMujica_DDD.domain.services.uncertainty_calculation_service import UncertaintyCalculationService

# --- Application Imports (reverted to absolute) ---
from AveMujica_DDD.application.services.forecast_service import ForecastService
from AveMujica_DDD.application.services.training_service import TrainingService
from AveMujica_DDD.application.ports.i_weather_data_provider import IWeatherDataProvider
from AveMujica_DDD.application.ports.i_prediction_engine import IPredictionEngine
from AveMujica_DDD.application.dtos.training_dto import TrainingRequestDTO

# --- Infrastructure Imports - 使用真实实现 ---
try:
    from AveMujica_DDD.infrastructure.repositories.file_system_repos import (
        FileForecastRepository,
        FileModelRepository,
        FileWeatherScenarioRepository
    )
    from AveMujica_DDD.infrastructure.data_providers.real_weather_provider import RealWeatherProvider
    from AveMujica_DDD.infrastructure.adapters.real_prediction_engine import RealPredictionEngine
    REAL_IMPLEMENTATIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Real implementations not available: {e}")
    REAL_IMPLEMENTATIONS_AVAILABLE = False
    # 设置为None，稍后会fallback到内存实现
    FileForecastRepository = None
    FileModelRepository = None
    FileWeatherScenarioRepository = None
    RealWeatherProvider = None
    RealPredictionEngine = None

# --- 保留内存仓储作为备选 ---
from AveMujica_DDD.infrastructure.repositories.in_memory_repos import (
    InMemoryForecastRepository,
    InMemoryModelRepository,
    InMemoryWeatherScenarioRepository
)


class DIContainer:
    """
    依赖注入容器，现在使用真实的实现组件。
    支持在真实组件和内存组件之间切换。
    """
    def __init__(self, use_real_implementations: bool = True):
        """
        初始化依赖注入容器。
        
        Args:
            use_real_implementations: 是否使用真实实现（True）或内存实现（False）
        """
        self.use_real_implementations = use_real_implementations
        
        if use_real_implementations:
            self._init_real_implementations()
        else:
            self._init_memory_implementations()
        
        # --- Domain Layer ---
        self.uncertainty_service: UncertaintyCalculationService = UncertaintyCalculationService()
        
        # 创建缺失的领域服务
        from AveMujica_DDD.domain.services.forecast_fusion_service import ForecastFusionService
        from AveMujica_DDD.domain.services.weather_scenario_recognition_service import WeatherScenarioRecognitionService
        
        self.fusion_service: ForecastFusionService = ForecastFusionService()
        self.scenario_recognition_service: WeatherScenarioRecognitionService = WeatherScenarioRecognitionService()

        # --- Application Layer ---
        self.forecast_service: ForecastService = ForecastService(
            forecast_repo=self.forecast_repo,
            model_repo=self.model_repo,
            weather_scenario_repo=self.weather_scenario_repo,
            weather_provider=self.weather_provider,
            prediction_engine=self.prediction_engine,
            uncertainty_service=self.uncertainty_service,
            fusion_service=self.fusion_service,
            scenario_recognition_service=self.scenario_recognition_service,
        )
        
        # 创建训练任务仓储（缺失的依赖）
        from AveMujica_DDD.domain.repositories.i_training_task_repository import ITrainingTaskRepository
        from AveMujica_DDD.infrastructure.repositories.in_memory_repos import InMemoryTrainingTaskRepository
        self.training_task_repo: ITrainingTaskRepository = InMemoryTrainingTaskRepository()
        
        self.training_service: TrainingService = TrainingService(
            training_task_repo=self.training_task_repo,
            model_repo=self.model_repo
        )

    def _init_real_implementations(self):
        """初始化真实的基础设施实现。"""
        if not REAL_IMPLEMENTATIONS_AVAILABLE:
            print("Real implementations not available, falling back to memory implementations")
            self._init_memory_implementations()
            return
            
        # --- Infrastructure Layer - Real Implementations ---
        try:
            self.forecast_repo: IForecastRepository = FileForecastRepository()
            self.model_repo: IModelRepository = FileModelRepository()
            self.weather_scenario_repo: IWeatherScenarioRepository = FileWeatherScenarioRepository()
            self.weather_provider: IWeatherDataProvider = RealWeatherProvider()
            self.prediction_engine: IPredictionEngine = RealPredictionEngine()
        except Exception as e:
            print(f"Failed to initialize real implementations: {e}")
            print("Falling back to memory implementations")
            self._init_memory_implementations()

    def _init_memory_implementations(self):
        """初始化内存实现（用于测试）。"""
        # --- Infrastructure Layer - Memory Implementations ---
        self.forecast_repo: IForecastRepository = InMemoryForecastRepository()
        self.model_repo: IModelRepository = InMemoryModelRepository()
        self.weather_scenario_repo: IWeatherScenarioRepository = InMemoryWeatherScenarioRepository()
        
        # 对于天气和预测，仍然可以使用真实实现
        self.weather_provider: IWeatherDataProvider = RealWeatherProvider()
        self.prediction_engine: IPredictionEngine = RealPredictionEngine()


def create_app(container: DIContainer) -> Flask:
    """
    应用工厂，用于创建Flask应用实例。
    """
    app = Flask(__name__)
    CORS(app)  # 允许跨域请求

    @app.route('/api/predict', methods=['POST'])
    def predict():
        """
        预测API端点，现在使用真实的数据和模型。
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid JSON"}), 400

            # --- 参数校验和转换 ---
            province = data.get('province')
            forecast_date_str = data.get('forecastDate')
            forecast_end_date_str = data.get('forecastEndDate', forecast_date_str)
            
            if not all([province, forecast_date_str]):
                return jsonify({"error": "Missing required parameters: province, forecastDate"}), 400

            try:
                start_date = datetime.strptime(forecast_date_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(forecast_end_date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({"error": "Invalid date format, expected YYYY-MM-DD"}), 400

            # --- 调用应用服务 ---
            # 尝试查找现有的训练模型
            model_id = None
            existing_models = container.model_repo.find_by_type_and_region(
                forecast_type=container.forecast_service._get_forecast_type_from_string('load'),
                region=province
            )
            
            if existing_models:
                # 使用找到的第一个模型
                model_id = existing_models[0].model_id
                print(f"Using existing model: {existing_models[0].name}")
            else:
                # 创建一个新的模型实例（在真实环境中应该是已训练的模型）
                model_id = uuid.uuid4()
                container.model_repo.seed_dummy_model(model_id, f'{province}_load_model')
                print(f"Created new model instance for {province}")
            
            forecast_dto = container.forecast_service.create_day_ahead_load_forecast(
                province=province,
                start_date=start_date,
                end_date=end_date,
                model_id=model_id
            )

            # --- 成功响应 ---
            response_data = {
                "success": True,
                "data": {
                    "forecast_id": str(forecast_dto.forecast_id),
                    "province": forecast_dto.province,
                    "model_name": forecast_dto.model_name,
                    "scenario": forecast_dto.scenario_type,
                    "predictions": [p.model_dump() for p in forecast_dto.time_series]
                },
                "cached": False,
                "implementation": "real" if container.use_real_implementations else "memory"
            }
            return jsonify(response_data)

        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 400
        except Exception as e:
            # 记录真实错误 for debugging
            print(f"An unexpected error occurred: {e}") 
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": "An internal server error occurred."}), 500

    @app.route('/api/provinces', methods=['GET'])
    def get_provinces():
        """
        获取支持的省份列表。
        """
        try:
            # 支持的省份列表
            provinces = ['上海', '江苏', '浙江', '安徽', '福建']
            
            return jsonify({
                "success": True,
                "data": provinces,
                "count": len(provinces)
            })
            
        except Exception as e:
            print(f"Error getting provinces: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/train', methods=['POST'])
    def train_model():
        """
        训练模型API端点。
        """
        try:
            data = request.get_json()
            print(f"Train API received data: {data}")  # 添加调试日志
            
            if not data:
                return jsonify({
                    "success": False,
                    "error": "Invalid JSON - no data received"
                }), 400

            # --- 参数校验 ---
            # 支持两种参数格式：下划线和驼峰
            province = data.get('province')
            forecast_type = data.get('forecastType') or data.get('forecast_type', 'load')
            start_date_str = data.get('trainStart') or data.get('train_start')
            end_date_str = data.get('trainEnd') or data.get('train_end')
            
            print(f"Extracted parameters: province={province}, forecast_type={forecast_type}, start={start_date_str}, end={end_date_str}")
            
            # 检查必需参数
            missing_params = []
            if not province:
                missing_params.append('province')
            if not start_date_str:
                missing_params.append('trainStart/train_start')
            if not end_date_str:
                missing_params.append('trainEnd/train_end')
                
            if missing_params:
                return jsonify({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing_params)}"
                }), 400

            # 日期格式验证
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                print(f"Parsed dates: start={start_date}, end={end_date}")
            except ValueError as ve:
                return jsonify({
                    "success": False,
                    "error": f"Invalid date format, expected YYYY-MM-DD: {str(ve)}"
                }), 400

            # 日期范围验证
            if start_date >= end_date:
                return jsonify({
                    "success": False,
                    "error": "Start date must be before end date"
                }), 400

            # --- 调用应用服务 ---
            task_id = str(uuid.uuid4())
            print(f"Starting training with task_id: {task_id}")
            
            # 启动训练（在真实环境中应该是异步的）
            training_result = container.training_service.train_model(
                province=province,
                forecast_type=forecast_type,
                start_date=start_date,
                end_date=end_date,
                task_id=task_id
            )
            
            print(f"Training completed: {training_result}")

            return jsonify({
                "success": True,
                "message": f"Training started for {province} {forecast_type} model",
                "task_id": task_id,
                "data": training_result
            })

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False, 
                "error": f"Training failed: {str(e)}"
            }), 500

    @app.route('/api/training-status/<task_id>', methods=['GET'])
    def get_training_status(task_id):
        """
        获取训练状态。
        """
        try:
            # 查询真实的训练任务状态
            status_dto = container.training_service.get_training_status(task_id)
            
            # 计算进度百分比
            progress = 100 if status_dto.status == "completed" else (
                50 if status_dto.status == "running" else 0
            )
            
            # 生成状态消息
            if status_dto.status == "completed":
                message = f"Training completed successfully. Model saved to: {status_dto.model_path}"
            elif status_dto.status == "failed":
                message = f"Training failed: {status_dto.error_message}"
            elif status_dto.status == "running":
                message = "Training in progress..."
            else:
                message = "Training pending..."
            
            return jsonify({
                "success": True,
                "data": {
                    "task_id": task_id,
                    "status": status_dto.status,
                    "progress": progress,
                    "message": message,
                    "model_type": status_dto.model_type,
                    "forecast_type": status_dto.forecast_type,
                    "province": status_dto.province,
                    "created_at": status_dto.created_at,
                    "started_at": status_dto.started_at,
                    "completed_at": status_dto.completed_at,
                    "model_path": status_dto.model_path,
                    "final_loss": status_dto.final_loss,
                    "mae": status_dto.mae,
                    "logs": status_dto.logs[-5:] if status_dto.logs else []  # 最近5条日志
                }
            })
            
        except Exception as e:
            print(f"Training status error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/historical-results', methods=['GET'])
    def get_historical_results():
        """
        获取历史预测结果。
        """
        try:
            # 获取查询参数
            model_type = request.args.get('modelType', 'load')
            prediction_type = request.args.get('predictionType', 'day_ahead')
            province = request.args.get('province', '上海')
            
            # 查询历史结果
            forecasts = container.forecast_repo.find_by_region(province)
            
            # 转换为前端需要的格式
            results = []
            for forecast in forecasts:
                results.append({
                    "forecast_id": str(forecast.forecast_id),
                    "province": forecast.province,
                    "model_name": forecast.model_name,
                    "scenario": forecast.scenario_type,
                    "created_at": forecast.created_at.isoformat(),
                    "predictions_count": len(forecast.time_series)
                })
            
            return jsonify({
                "success": True,
                "data": results,
                "count": len(results)
            })
            
        except Exception as e:
            print(f"Historical results error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/clear-cache', methods=['POST'])
    def clear_cache():
        """
        清除缓存。
        """
        try:
            # 在真实环境中，这应该清除实际的缓存
            # 这里只是模拟清除缓存的响应
            
            return jsonify({
                "success": True,
                "message": "Cache cleared successfully",
                "cleared_items": 0
            })
            
        except Exception as e:
            print(f"Clear cache error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/cache-stats', methods=['GET'])
    def get_cache_stats():
        """
        获取缓存统计信息。
        """
        try:
            # 在真实环境中，这应该返回实际的缓存统计信息
            stats = {
                "total_entries": 0,
                "hit_rate": 0.0,
                "miss_rate": 0.0,
                "memory_usage": "0 MB"
            }
            
            return jsonify({
                "success": True,
                "data": stats
            })
            
        except Exception as e:
            print(f"Cache stats error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """
        健康检查端点。
        """
        return jsonify({
            "status": "healthy",
            "implementation": "real" if container.use_real_implementations else "memory",
            "timestamp": datetime.now().isoformat()
        })

    @app.route('/api/models', methods=['GET'])
    def list_models():
        """
        列出所有可用的模型。
        """
        try:
            models = container.model_repo.list_all()
            model_list = [
                {
                    "model_id": str(model.model_id),
                    "name": model.name,
                    "forecast_type": model.forecast_type.value,
                    "file_path": model.file_path
                }
                for model in models
            ]
            
            return jsonify({
                "success": True,
                "models": model_list,
                "count": len(model_list)
            })
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/scenarios', methods=['GET'])
    def list_scenarios():
        """
        列出所有天气场景。
        """
        try:
            scenarios = container.weather_scenario_repo.list_all()
            scenario_list = [
                {
                    "scenario_type": scenario.scenario_type.value,
                    "description": scenario.description,
                    "uncertainty_multiplier": scenario.uncertainty_multiplier
                }
                for scenario in scenarios
            ]
            
            return jsonify({
                "success": True,
                "scenarios": scenario_list,
                "count": len(scenario_list)
            })
            
        except Exception as e:
            print(f"Error listing scenarios: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/interval-forecast', methods=['POST'])
    def interval_forecast():
        """
        区间预测API端点。
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "Invalid JSON"}), 400

            # 参数解析
            province = data.get('province')
            forecast_date_str = data.get('forecastDate')
            forecast_type = data.get('forecastType', 'load')
            confidence_level = data.get('confidenceLevel', 0.95)
            
            if not all([province, forecast_date_str]):
                return jsonify({
                    "success": False, 
                    "error": "Missing required parameters: province, forecastDate"
                }), 400

            try:
                start_date = datetime.strptime(forecast_date_str, '%Y-%m-%d').date()
                end_date = start_date  # 单日预测
            except ValueError:
                return jsonify({
                    "success": False, 
                    "error": "Invalid date format, expected YYYY-MM-DD"
                }), 400

            # 调用区间预测服务
            from AveMujica_DDD.domain.aggregates.prediction_model import ForecastType
            
            forecast_type_enum = container.forecast_service._get_forecast_type_from_string(forecast_type)
            
            forecast_dto = container.forecast_service.create_interval_forecast(
                province=province,
                start_date=start_date,
                end_date=end_date,
                forecast_type=forecast_type_enum,
                confidence_level=confidence_level
            )

            response_data = {
                "success": True,
                "data": {
                    "forecast_id": str(forecast_dto.forecast_id),
                    "province": forecast_dto.province,
                    "model_name": forecast_dto.model_name,
                    "scenario": forecast_dto.scenario_type,
                    "confidence_level": confidence_level,
                    "predictions": [p.model_dump() for p in forecast_dto.time_series]
                }
            }
            return jsonify(response_data)

        except Exception as e:
            print(f"Interval forecast error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/recognize-scenarios', methods=['POST'])
    def recognize_scenarios():
        """
        天气场景识别API端点。
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "Invalid JSON"}), 400

            # 参数解析
            weather_data = data.get('weatherData', {})
            
            # 模拟场景识别（实际应用中会使用真实的识别算法）
            scenario_result = {
                "recognized_scenario": "温和正常",
                "confidence": 0.85,
                "features": {
                    "temperature": weather_data.get('temperature', 20),
                    "humidity": weather_data.get('humidity', 60),
                    "wind_speed": weather_data.get('wind_speed', 4),
                    "precipitation": weather_data.get('precipitation', 0)
                },
                "uncertainty_multiplier": 1.0,
                "description": "温和正常天气条件，电力系统运行稳定"
            }

            return jsonify({
                "success": True,
                "data": scenario_result
            })

        except Exception as e:
            print(f"Scenario recognition error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    return app


# --- 创建默认的应用实例供导入使用 ---
try:
    # 默认使用内存实现，避免在导入时依赖外部文件
    default_container = DIContainer(use_real_implementations=False)
    app = create_app(default_container)
except Exception as e:
    print(f"Warning: Failed to create default app instance: {e}")
    app = None


if __name__ == '__main__':
    # --- 启动应用 ---
    # 可以通过环境变量控制使用真实实现还是内存实现
    import os
    use_real = os.getenv('USE_REAL_IMPLEMENTATIONS', 'true').lower() == 'true'
    
    di_container = DIContainer(use_real_implementations=use_real)
    app = create_app(di_container)
    
    print(f"Starting application with {'REAL' if use_real else 'MEMORY'} implementations")
    app.run(debug=True, port=5001) 