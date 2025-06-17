# 🐳 Docker部署指南

## 项目概述
这是一个基于深度学习的多源电力预测与调度平台，支持负荷、光伏、风电预测的Docker化部署方案。

## 📋 部署前准备

### Windows系统要求
- Windows 10/11 (推荐)
- 至少8GB内存 (推荐16GB)
- 至少20GB可用磁盘空间
- 稳定的网络连接

### 安装环境依赖

#### 1. 安装Docker Desktop
```powershell
# 下载并安装Docker Desktop for Windows
# https://docs.docker.com/desktop/install/windows-install/

# 验证安装
docker --version
docker-compose --version
```

#### 2. 启用WSL2 (推荐)
```powershell
# 管理员权限运行PowerShell
wsl --install
# 重启计算机后设置WSL2为默认
wsl --set-default-version 2
```

## 🚀 快速部署步骤

### 步骤1: 准备项目文件
```powershell
# 1. 将整个项目文件夹复制到新的Windows环境
# 2. 进入项目目录
cd "你的项目路径"

# 3. 确认重要文件存在
dir Dockerfile
dir docker-compose.yml
dir requirements.txt
```

### 步骤2: 构建和启动应用
```powershell
# 方式1: 使用docker-compose (推荐)
docker-compose up --build -d

# 方式2: 手动构建和运行
docker build -t scene-forecast .
docker run -d -p 5000:5000 --name scene_forecast_app scene-forecast
```

### 步骤3: 验证部署
```powershell
# 检查容器状态
docker ps

# 查看容器日志
docker-compose logs -f

# 测试API接口
curl http://localhost:5000/api/health
# 或在浏览器中访问: http://localhost:5000/api/health
```

## 🔧 配置说明

### 数据持久化
以下目录会自动挂载到宿主机，确保数据不丢失：
- `./data` - 训练和预测数据
- `./results` - 预测结果
- `./logs` - 应用日志
- `./models` - 训练好的模型

### 端口配置
- 应用端口: `5000` (映射到宿主机5000端口)
- 如需修改，编辑 `docker-compose.yml` 中的 `ports` 配置

### 资源配置
默认配置：
- CPU限制: 2核
- 内存限制: 4GB
- 内存预留: 2GB

可在 `docker-compose.yml` 中调整。

## 🛠 常用管理命令

### 启动和停止
```powershell
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看实时日志
docker-compose logs -f scene-forecast
```

### 维护命令
```powershell
# 进入容器调试
docker exec -it scene_forecast_app bash

# 清理未使用的镜像和容器
docker system prune -a

# 查看资源使用情况
docker stats

# 备份数据
docker run --rm -v scene_forecast_data:/data -v ${PWD}:/backup alpine tar czf /backup/backup.tar.gz -C /data .
```

## 🔧 故障排除

### 常见问题

#### 1. 端口被占用
```powershell
# 查看端口占用
netstat -ano | findstr :5000

# 修改docker-compose.yml中的端口映射
ports:
  - "5001:5000"  # 改为5001端口
```

#### 2. 内存不足
```powershell
# 检查系统资源
wmic computersystem get TotalPhysicalMemory
docker stats

# 调整内存配置
# 在docker-compose.yml中减少内存限制
limits:
  memory: 2G
```

#### 3. 容器启动失败
```powershell
# 查看详细错误信息
docker-compose logs scene-forecast

# 检查配置文件语法
docker-compose config

# 重新构建镜像
docker-compose build --no-cache
```

## 📊 监控和日志

### 日志查看
```powershell
# 实时查看所有日志
docker-compose logs -f

# 查看最近100行日志
docker-compose logs --tail=100 scene-forecast

# 查看特定时间的日志
docker-compose logs --since="2024-01-01T00:00:00" scene-forecast
```

### 性能监控
```powershell
# 查看资源使用情况
docker stats scene_forecast_app

# 查看容器详细信息
docker inspect scene_forecast_app
```

## 🌐 访问应用

部署成功后，可以通过以下方式访问：

- **API接口**: http://localhost:5000/api/
- **健康检查**: http://localhost:5000/api/health
- **前端界面**: http://localhost:5000/ (如果有前端)

## 📞 技术支持

如果遇到问题，请提供以下信息：
1. 错误日志
2. 系统信息
3. 容器状态
4. 具体错误描述