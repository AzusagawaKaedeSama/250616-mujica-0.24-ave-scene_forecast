@echo off
chcp 65001 >nul
echo ===========================================
echo    深度学习电力预测系统 - Docker一键部署
echo ===========================================
echo.

REM 检查Docker是否安装
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker未安装或未启动
    echo 请先安装Docker Desktop for Windows
    pause
    exit /b 1
)

REM 检查docker-compose是否可用
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] docker-compose未安装或不可用
    pause
    exit /b 1
)

echo [信息] Docker环境检查通过
echo.

REM 检查必要文件
if not exist "Dockerfile" (
    echo [错误] 未找到Dockerfile文件
    pause
    exit /b 1
)

if not exist "docker-compose.yml" (
    echo [错误] 未找到docker-compose.yml文件
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo [错误] 未找到requirements.txt文件
    pause
    exit /b 1
)

echo [信息] 项目文件检查通过
echo.

REM 创建必要的目录
if not exist "logs" mkdir logs
if not exist "results" mkdir results
if not exist "models" mkdir models
echo [信息] 目录结构检查完成
echo.

REM 停止并清理旧容器（如果存在）
echo [信息] 停止旧容器...
docker-compose down 2>nul
echo.

REM 构建和启动容器
echo [信息] 开始构建Docker镜像（首次构建可能需要10-30分钟）...
echo [信息] 请耐心等待...
echo.

docker-compose up --build -d

if %errorlevel% equ 0 (
    echo.
    echo ===========================================
    echo             部署成功！
    echo ===========================================
    echo.
    echo 应用已成功启动，可通过以下方式访问：
    echo.
    echo  🌐 应用主页: http://localhost:5000
    echo  ⚡ 健康检查: http://localhost:5000/api/health
    echo  📊 API接口: http://localhost:5000/api/
    echo.
    echo 常用管理命令：
    echo  查看日志: docker-compose logs -f
    echo  停止服务: docker-compose down
    echo  重启服务: docker-compose restart
    echo.
    echo 正在检查服务状态...
    timeout /t 10 /nobreak >nul
    
    REM 检查服务是否正常启动
    curl -s http://localhost:5000/api/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo [成功] 服务已正常启动并响应请求
    ) else (
        echo [提示] 服务正在启动中，请稍后检查日志
        echo 查看启动日志: docker-compose logs -f scene-forecast
    )
    
) else (
    echo.
    echo ===========================================
    echo             部署失败！
    echo ===========================================
    echo.
    echo 请检查错误信息并尝试以下操作：
    echo 1. 查看详细日志: docker-compose logs
    echo 2. 检查端口占用: netstat -ano ^| findstr :5000
    echo 3. 重新构建: docker-compose build --no-cache
    echo 4. 联系技术支持
)

echo.
pause 