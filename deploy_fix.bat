@echo off
chcp 65001
echo ========================================
echo 多源电力预测系统 - 网络优化部署脚本
echo ========================================

echo.
echo [1/6] 检查Docker环境...
docker --version
if %ERRORLEVEL% neq 0 (
    echo 错误：Docker未安装或未启动！
    echo 请先启动Docker Desktop
    pause
    exit /b 1
)

echo.
echo [2/6] 清理旧的容器和镜像...
docker-compose down 2>nul
docker system prune -f

echo.
echo [3/6] 配置网络和DNS...
docker network ls | findstr bridge
if %ERRORLEVEL% neq 0 (
    echo 创建Docker网络...
    docker network create --driver bridge scene_forecast_network
)

echo.
echo [4/6] 尝试预拉取镜像（使用多个镜像源）...
echo 正在尝试拉取Python镜像...
docker pull python:3.9-slim
if %ERRORLEVEL% neq 0 (
    echo 主镜像源失败，尝试其他方式...
    echo 如果此步骤失败，请检查网络连接或配置镜像加速器
)

echo.
echo [5/6] 开始构建和部署...
docker-compose up --build -d

echo.
echo [6/6] 验证部署状态...
timeout /t 10 /nobreak >nul
docker ps -a

echo.
echo ========================================
echo 部署完成！
echo ========================================
echo.
echo 应用访问地址:
echo   主页: http://localhost:5000
echo   健康检查: http://localhost:5000/api/health
echo.
echo 常用管理命令:
echo   查看日志: docker-compose logs -f
echo   停止服务: docker-compose down
echo   重启服务: docker-compose restart
echo.
echo 如果部署失败，请：
echo 1. 确保Docker Desktop正在运行
echo 2. 检查网络连接
echo 3. 配置Docker镜像加速器
echo 4. 运行: docker-compose logs 查看详细错误
echo.
pause 