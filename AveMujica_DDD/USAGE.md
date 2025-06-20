# MUJICA DDD系统使用指南

## 🚀 快速启动

### 1. 启动API服务器（推荐）

**适用于Web界面用户:**

```bash
# 激活环境
conda activate tf_gpu

# 启动API服务器
python -m AveMujica_DDD.quick_start_api
```

服务器将在 `http://localhost:5001` 启动，前端Web界面可以连接到这个API。

### 2. 完整系统启动（交互式）

```bash
# 激活环境
conda activate tf_gpu

# 启动完整系统
python -m AveMujica_DDD.start_system
```

这将提供交互式选项：
- 完整演示（预测演示 + API服务器）
- 仅启动API服务器
- 仅运行预测演示
- 退出

### 3. 程序化使用

```python
from AveMujica_DDD.main import MujicaDDDSystem

# 初始化系统
system = MujicaDDDSystem()

# 执行预测
result = system.run_example_forecast(province="上海", days_ahead=1)

# 启动API服务器
system.start_api_server(port=5001)
```

## 🌐 Web界面使用

### 前端连接

确保前端配置连接到正确的API端点：
- API基础URL: `http://localhost:5001`
- 健康检查: `GET /api/health`

### 训练模型

通过Web界面训练新模型：

1. **选择参数**：
   - 省份：上海、江苏、浙江、安徽、福建
   - 预测类型：load（负荷）、pv（光伏）、wind（风电）
   - 训练日期范围：如 2024-01-01 到 2024-08-31

2. **点击训练按钮**：
   - 系统将返回训练任务ID
   - 自动轮询训练状态
   - 训练完成后显示结果

### 执行预测

通过Web界面执行预测：

1. **选择参数**：
   - 省份：支持的省份列表
   - 预测日期：YYYY-MM-DD格式
   - 预测类型：日前预测、区间预测等

2. **获取结果**：
   - 预测时间序列数据
   - 置信区间（如果选择区间预测）
   - 天气场景信息
   - 不确定性分析

## 📊 API端点详解

### 核心预测API

**POST /api/predict**
```json
{
  "province": "上海",
  "forecastDate": "2025-06-20",
  "forecastEndDate": "2025-06-20"
}
```

**响应:**
```json
{
  "success": true,
  "data": {
    "forecast_id": "uuid",
    "province": "上海", 
    "model_name": "上海_load_convtrans_weather",
    "scenario": "温和正常",
    "predictions": [
      {
        "timestamp": "2025-06-20T00:00:00",
        "value": 25348.2,
        "lower_bound": 24080.8,
        "upper_bound": 26615.5
      }
    ]
  }
}
```

### 模型训练API

**POST /api/train**
```json
{
  "province": "上海",
  "forecast_type": "load",
  "train_start": "2024-01-01", 
  "train_end": "2024-08-31"
}
```

**响应:**
```json
{
  "success": true,
  "task_id": "uuid",
  "message": "Training started for 上海 load model"
}
```

### 其他API

- `GET /api/health` - 健康检查
- `GET /api/models` - 获取所有模型列表
- `GET /api/provinces` - 获取支持的省份列表
- `GET /api/scenarios` - 获取天气场景列表
- `GET /api/historical-results` - 获取历史预测结果

## 🏗️ 系统架构

### DDD四层架构

```
┌─────────────────────────────────────────┐
│     用户接口层 (Interface Layer)          │  ← Web界面, API
├─────────────────────────────────────────┤
│     应用层 (Application Layer)           │  ← 用例编排
├─────────────────────────────────────────┤
│     领域层 (Domain Layer)               │  ← 业务逻辑
├─────────────────────────────────────────┤
│     基础设施层 (Infrastructure Layer)    │  ← 数据存储, ML模型
└─────────────────────────────────────────┘
```

### 核心组件

1. **预测聚合 (Forecast)**：预测结果的完整表示
2. **模型聚合 (PredictionModel)**：机器学习模型的业务抽象
3. **天气场景聚合 (WeatherScenario)**：17种天气场景类型
4. **预测服务 (ForecastService)**：核心预测用例编排
5. **训练服务 (TrainingService)**：模型训练用例编排

## 🔧 已有模型集成

系统能够自动发现并集成您现有的训练模型：

### 模型目录结构
```
models/
└── convtrans_weather/
    ├── load/
    │   ├── 上海/
    │   │   ├── best_model.pth
    │   │   ├── weather_config.json
    │   │   └── ...
    │   └── 江苏/
    └── pv/
        └── wind/
```

### 自动模型注册

- 系统启动时自动扫描模型目录
- 为每个发现的模型创建聚合对象
- 注册到模型仓储中
- 在Web界面中可选择使用

## ⚠️  故障排除

### 常见问题

1. **模型加载失败**：
   - 检查模型文件路径
   - 确认PyTorch模型格式兼容性
   - 系统会自动降级到合成预测

2. **训练API返回400错误**：
   - 检查参数格式（支持下划线和驼峰格式）
   - 确认日期格式为YYYY-MM-DD
   - 查看控制台日志获取详细错误信息

3. **前端连接错误**：
   - 确认API服务器已启动（端口5001）
   - 检查CORS设置
   - 确认前端API配置正确

### 调试模式

启动调试模式获取详细日志：

```bash
python -m AveMujica_DDD.api
```

查看控制台输出了解详细的错误信息和执行流程。

## 🔄 从旧系统迁移

### 完全替代

重构后的DDD系统可以完全替代原始系统：

- ✅ **更简单的启动**：单一命令启动
- ✅ **更稳定的API**：统一的错误处理
- ✅ **更好的扩展性**：模块化架构
- ✅ **更易维护**：清晰的层次边界

### 保持兼容

- 现有训练模型自动集成
- API接口保持兼容
- 预测结果格式一致
- 支持所有原有功能

## 🎯 最佳实践

1. **生产环境**：使用 `quick_start_api.py` 启动API服务器
2. **开发调试**：使用 `start_system.py` 的交互式模式
3. **模型训练**：通过Web界面进行，避免命令行操作
4. **错误处理**：系统会优雅降级，确保服务可用性
5. **性能优化**：系统自动缓存模型，减少重复加载时间 