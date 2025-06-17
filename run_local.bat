@echo off
chcp 65001
echo ==========================================
echo 多源电力预测系统 - 本地Python部署脚本
echo ==========================================

echo.
echo [1/6] 检查Python环境...
python --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Python未安装！
    echo.
    echo 请先安装Python 3.9：
    echo 1. 访问：https://www.python.org/downloads/
    echo 2. 下载Python 3.9.x版本
    echo 3. 安装时勾选"Add Python to PATH"
    echo.
    echo 或者在管理员PowerShell中运行：
    echo winget install Python.Python.3.9
    echo.
    pause
    exit /b 1
)

echo ✅ Python环境检查通过
python --version
pip --version

echo.
echo [2/6] 检查虚拟环境...
if not exist "venv\" (
    echo 📦 创建虚拟环境...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo ❌ 虚拟环境创建失败！
        pause
        exit /b 1
    )
    echo ✅ 虚拟环境创建成功
) else (
    echo ✅ 虚拟环境已存在
)

echo.
echo [3/6] 激活虚拟环境...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ❌ 虚拟环境激活失败！
    pause
    exit /b 1
)
echo ✅ 虚拟环境已激活

echo.
echo [4/6] 升级pip...
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

echo.
echo [5/6] 安装项目依赖...
echo 正在安装依赖包，请稍候...
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if %ERRORLEVEL% neq 0 (
    echo ❌ 依赖安装失败！请检查requirements.txt文件
    pause
    exit /b 1
)
echo ✅ 依赖安装完成

echo.
echo [6/6] 启动应用...
echo ==========================================
echo 🚀 正在启动多源电力预测系统...
echo ==========================================
echo.
echo 应用启动后可通过以下地址访问：
echo   主页: http://localhost:5000
echo   健康检查: http://localhost:5000/api/health
echo   API文档: http://localhost:5000/api/
echo.
echo 按 Ctrl+C 可停止应用
echo ==========================================

python app.py

echo.
echo 应用已停止运行
pause 