import os
import json
import uuid
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import pandas as pd

from AveMujica_DDD.domain.aggregates.forecast import Forecast
from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
from AveMujica_DDD.domain.aggregates.weather_scenario import WeatherScenario, ScenarioType
from AveMujica_DDD.domain.repositories.i_forecast_repository import IForecastRepository
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository
from AveMujica_DDD.domain.repositories.i_weather_scenario_repository import IWeatherScenarioRepository


class FileForecastRepository(IForecastRepository):
    def __init__(self, base_dir: str = 'results'):
        self.base_dir = base_dir
        self.forecasts_dir = os.path.join(base_dir, 'forecasts')
        os.makedirs(self.forecasts_dir, exist_ok=True)
        print(f"FileForecastRepository initialized with directory: {self.forecasts_dir}")

    def save(self, forecast: Forecast) -> None:
        try:
            date_str = forecast.creation_time.strftime('%Y-%m-%d')
            province_dir = os.path.join(self.forecasts_dir, forecast.province, date_str)
            os.makedirs(province_dir, exist_ok=True)
            
            timestamp_str = forecast.creation_time.strftime('%Y%m%d_%H%M%S')
            base_filename = f"forecast_{timestamp_str}_{forecast.forecast_id}"
            
            json_file = os.path.join(province_dir, f"{base_filename}.json")
            self._save_forecast_json(forecast, json_file)
            
            print(f"Forecast {forecast.forecast_id} saved to {province_dir}")
        except Exception as e:
            print(f"Error saving forecast {forecast.forecast_id}: {e}")

    def find_by_id(self, forecast_id: uuid.UUID) -> Optional[Forecast]:
        return None

    def find_by_province_and_date(self, province: str, target_date: date) -> List[Forecast]:
        return []

    def list_all(self) -> List[Forecast]:
        return []

    def _save_forecast_json(self, forecast: Forecast, filepath: str) -> None:
        data = {
            'forecast_id': str(forecast.forecast_id),
            'province': forecast.province,
            'creation_time': forecast.creation_time.isoformat(),
            'model_name': forecast.prediction_model.name,
            'model_type': forecast.prediction_model.forecast_type.value,
            'scenario_type': forecast.matched_weather_scenario.scenario_type.value,
            'time_series': [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'lower_bound': point.lower_bound,
                    'upper_bound': point.upper_bound
                }
                for point in forecast.time_series
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class FileModelRepository(IModelRepository):
    def __init__(self, base_dir: str = 'models'):
        self.base_dir = base_dir
        self.models_index_file = os.path.join(base_dir, 'models_index.json')
        os.makedirs(base_dir, exist_ok=True)
        self.models_index = self._load_models_index()
        
        # 自动发现现有模型文件
        self._discover_existing_models()
        
        print(f"FileModelRepository initialized with directory: {base_dir}")
        print(f"Found {len(self.models_index)} registered models")

    def _discover_existing_models(self):
        """自动发现现有的模型文件并注册到索引中"""
        convtrans_weather_dir = os.path.join(self.base_dir, 'convtrans_weather')
        if not os.path.exists(convtrans_weather_dir):
            return
            
        # 扫描三种预测类型的目录
        for forecast_type in ['load', 'pv', 'wind']:
            type_dir = os.path.join(convtrans_weather_dir, forecast_type)
            if not os.path.exists(type_dir):
                continue
                
            # 扫描每个省份目录
            for province in os.listdir(type_dir):
                province_dir = os.path.join(type_dir, province)
                if not os.path.isdir(province_dir):
                    continue
                    
                # 检查是否有模型文件
                model_files = ['best_model.pth', 'convtrans_weather_model.pth']
                model_file_path = None
                
                for model_file in model_files:
                    full_path = os.path.join(province_dir, model_file)
                    if os.path.exists(full_path):
                        model_file_path = full_path
                        break
                
                if model_file_path:
                    # 创建模型索引条目
                    model_name = f"{province}_{forecast_type}_convtrans_weather"
                    model_id = self._generate_model_id_from_path(model_file_path)
                    
                    # 检查是否已经注册
                    if str(model_id) not in self.models_index:
                        self._register_discovered_model(
                            model_id=model_id,
                            name=model_name,
                            forecast_type=forecast_type.upper(),
                            province=province,
                            file_path=model_file_path
                        )

    def _generate_model_id_from_path(self, file_path: str) -> uuid.UUID:
        """从文件路径生成一致的UUID"""
        import hashlib
        hash_obj = hashlib.md5(file_path.encode())
        return uuid.UUID(hash_obj.hexdigest())

    def _register_discovered_model(self, model_id: uuid.UUID, name: str, 
                                 forecast_type: str, province: str, file_path: str):
        """注册发现的模型"""
        # 尝试读取配置文件以获取更多信息
        config_file = os.path.join(os.path.dirname(file_path), 'weather_config.json')
        feature_columns = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        target_column = forecast_type.lower()
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    feature_columns = config.get('weather_features', feature_columns)
            except:
                pass
        
        model_info = {
            'model_id': str(model_id),
            'name': name,
            'version': '1.0.0',
            'forecast_type': forecast_type,
            'file_path': file_path,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'description': f'{province} {forecast_type} 天气感知预测模型',
            'region': province,
            'creation_time': datetime.now().isoformat(),
            'is_discovered': True  # 标记为自动发现的模型
        }
        
        self.models_index[str(model_id)] = model_info
        self._save_models_index()
        print(f"Discovered and registered model: {name}")

    def find_by_id(self, model_id: uuid.UUID) -> Optional[PredictionModel]:
        model_id_str = str(model_id)
        if model_id_str in self.models_index:
            model_info = self.models_index[model_id_str]
            return self._create_model_from_info(model_info)
        return None

    def find_by_type_and_region(self, forecast_type: ForecastType, region: str) -> List[PredictionModel]:
        models = []
        for model_info in self.models_index.values():
            if (model_info['forecast_type'] == forecast_type.value and 
                model_info.get('region') == region):
                model = self._create_model_from_info(model_info)
                if model:
                    models.append(model)
        return models

    def save(self, model: PredictionModel) -> None:
        # 从模型名称中提取region信息
        region = model.name.split('_')[0] if '_' in model.name else 'unknown'
        
        model_info = {
            'model_id': str(model.model_id),
            'name': model.name,
            'version': model.version,
            'forecast_type': model.forecast_type.value,
            'file_path': model.file_path,
            'target_column': model.target_column,
            'feature_columns': model.feature_columns,
            'description': model.description,
            'region': region,
            'creation_time': datetime.now().isoformat()
        }
        
        self.models_index[str(model.model_id)] = model_info
        self._save_models_index()
        print(f"Model {model.name} saved to repository (region: {region})")

    def list_all(self) -> List[PredictionModel]:
        models = []
        for model_info in self.models_index.values():
            model = self._create_model_from_info(model_info)
            if model:
                models.append(model)
        return models

    def seed_dummy_model(self, model_id: uuid.UUID, model_name: str) -> None:
        # 提取省份名称用作region
        region = model_name.split('_')[0] if '_' in model_name else 'unknown'
        
        model_info = {
            'model_id': str(model_id),
            'name': model_name,
            'version': '1.0.0',
            'forecast_type': ForecastType.LOAD.value,  # 使用枚举值
            'file_path': f'dummy/{model_name}.pth',
            'target_column': 'load',
            'feature_columns': ['temperature', 'humidity'],
            'description': f'虚拟{region}负荷预测模型',
            'creation_time': datetime.now().isoformat(),
            'region': region
        }
        
        self.models_index[str(model_id)] = model_info
        self._save_models_index()  # 保存到文件
        print(f"Dummy model {model_name} seeded in file repository (region: {region})")

    def _load_models_index(self) -> Dict[str, Any]:
        if os.path.exists(self.models_index_file):
            try:
                with open(self.models_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_models_index(self) -> None:
        try:
            with open(self.models_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.models_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving models index: {e}")

    def _create_model_from_info(self, model_info: Dict[str, Any]) -> Optional[PredictionModel]:
        try:
            return PredictionModel(
                model_id=uuid.UUID(model_info['model_id']),
                name=model_info['name'],
                version=model_info.get('version', '1.0.0'),
                forecast_type=ForecastType(model_info['forecast_type']),
                file_path=model_info['file_path'],
                target_column=model_info['target_column'],
                feature_columns=model_info['feature_columns'],
                description=model_info.get('description', '')
            )
        except Exception as e:
            print(f"Error creating model from info: {e}")
            return None


class FileWeatherScenarioRepository(IWeatherScenarioRepository):
    def __init__(self, base_dir: str = 'scenarios'):
        self.base_dir = base_dir
        self.scenarios_file = os.path.join(base_dir, 'weather_scenarios.json')
        os.makedirs(base_dir, exist_ok=True)
        self.scenarios = self._load_scenarios()
        print(f"FileWeatherScenarioRepository initialized with {len(self.scenarios)} scenarios")

    def find_by_type(self, scenario_type: ScenarioType) -> Optional[WeatherScenario]:
        for scenario in self.scenarios.values():
            if scenario.scenario_type == scenario_type:
                return scenario
        return None

    def save(self, scenario: WeatherScenario) -> None:
        self.scenarios[scenario.scenario_type.value] = scenario
        self._save_scenarios()

    def list_all(self) -> List[WeatherScenario]:
        return list(self.scenarios.values())

    def _load_scenarios(self) -> Dict[str, WeatherScenario]:
        scenarios = {}
        
        if os.path.exists(self.scenarios_file):
            try:
                with open(self.scenarios_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for scenario_data in data:
                    scenario = WeatherScenario(
                        scenario_type=ScenarioType(scenario_data['scenario_type']),
                        description=scenario_data['description'],
                        uncertainty_multiplier=scenario_data['uncertainty_multiplier']
                    )
                    scenarios[scenario.scenario_type.value] = scenario
            except Exception:
                pass
        
        if not scenarios:
            scenarios = self._create_default_scenarios()
            self._save_scenarios()
        
        return scenarios

    def _save_scenarios(self) -> None:
        try:
            data = []
            for scenario in self.scenarios.values():
                data.append({
                    'scenario_type': scenario.scenario_type.value,
                    'description': scenario.description,
                    'uncertainty_multiplier': scenario.uncertainty_multiplier
                })
            
            with open(self.scenarios_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _create_default_scenarios(self) -> Dict[str, WeatherScenario]:
        """创建默认的17种天气场景"""
        scenarios = {}
        
        # 极端天气场景(6种)
        extreme_scenarios = [
            (ScenarioType.EXTREME_STORM_RAIN, "极端暴雨天气场景，降水量>50mm/h", 3.5),
            (ScenarioType.EXTREME_HOT_HUMID, "极端高温高湿天气场景，温度>35°C，湿度>85%", 3.0),
            (ScenarioType.EXTREME_STRONG_WIND, "极端大风天气场景，风速>20m/s", 2.8),
            (ScenarioType.EXTREME_HEAVY_RAIN, "特大暴雨天气场景，降水量>100mm/h", 4.0),
            (ScenarioType.EXTREME_HOT, "极端高温天气场景，温度>40°C", 2.5),
            (ScenarioType.EXTREME_COLD, "极端寒冷天气场景，温度<-10°C", 3.0),
        ]
        
        # 典型场景(3种)
        typical_scenarios = [
            (ScenarioType.TYPICAL_GENERAL_NORMAL, "一般正常天气场景，负荷平稳", 1.2),
            (ScenarioType.TYPICAL_RAINY_LOW_LOAD, "多雨低负荷天气场景，降水适中", 1.1),
            (ScenarioType.TYPICAL_MILD_HUMID_HIGH_LOAD, "温和高湿高负荷天气场景", 1.8),
        ]
        
        # 普通天气变种(4种)
        normal_scenarios = [
            (ScenarioType.NORMAL_SPRING_MILD, "春季温和天气场景，温度15-25°C", 0.9),
            (ScenarioType.NORMAL_SUMMER_COMFORTABLE, "夏季舒适天气场景，温度25-30°C", 1.0),
            (ScenarioType.NORMAL_AUTUMN_STABLE, "秋季平稳天气场景，温度10-20°C", 0.8),
            (ScenarioType.NORMAL_WINTER_MILD, "冬季温和天气场景，温度0-15°C", 1.3),
        ]
        
        # 基础场景(4种)
        basic_scenarios = [
            (ScenarioType.HIGH_WIND_SUNNY, "大风晴朗天气场景，风速10-15m/s", 1.1),
            (ScenarioType.CALM_CLOUDY, "无风阴天天气场景，风速<3m/s", 0.9),
            (ScenarioType.MODERATE_NORMAL, "温和正常天气场景，标准基准", 1.0),
            (ScenarioType.STORM_RAIN, "暴雨大风天气场景，综合影响", 2.2),
        ]
        
        # 创建所有场景
        all_scenarios = extreme_scenarios + typical_scenarios + normal_scenarios + basic_scenarios
        
        for scenario_type, description, multiplier in all_scenarios:
            scenario = WeatherScenario(
                scenario_type=scenario_type,
                description=description,
                uncertainty_multiplier=multiplier
            )
            scenarios[scenario.scenario_type.value] = scenario
        
        return scenarios