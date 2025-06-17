@echo off
chcp 65001
echo ==========================================
echo 多源电力预测系统 - 完整项目启动脚本
echo ==========================================

echo.
echo 📋 启动前检查...

echo.
echo [1/4] 检查Python环境...
python --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Python未安装！请先运行: install_python.bat
    pause
    exit /b 1
)
echo ✅ Python环境正常

echo.
echo [2/4] 检查Node.js环境...
node --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Node.js未安装！请先运行: install_nodejs.bat
    pause
    exit /b 1
)
echo ✅ Node.js环境正常

echo.
echo [3/4] 检查前端依赖...
if not exist "frontend\node_modules\" (
    echo ⚠️  前端依赖未安装，正在自动安装...
    cd frontend
    npm config set registry https://registry.npmmirror.com
    npm install
    if %ERRORLEVEL% neq 0 (
        echo ❌ 前端依赖安装失败！请手动运行: setup_frontend.bat
        pause
        exit /b 1
    )
    cd ..
)
echo ✅ 前端依赖已安装

echo.
echo [4/4] 启动项目...
echo ==========================================
echo 🚀 正在启动完整项目...
echo ==========================================

echo.
echo 📍 启动顺序：
echo   1️⃣ 后端服务 (Python Flask) - 端口 5000
echo   2️⃣ 前端服务 (React Vite) - 端口 5173
echo.

echo 🔧 启动后端服务...
start "后端服务" cmd /k "call venv\Scripts\activate.bat && python app.py"

echo ⏳ 等待后端启动...
timeout /t 5 /nobreak >nul

echo 🌐 启动前端服务...
start "前端服务" cmd /k "cd frontend && npm run dev"

echo.
echo ==========================================
echo ✅ 项目启动完成！
echo ==========================================
echo.
echo 📱 访问地址：
echo   🌐 前端界面: http://localhost:5173
echo   ⚡ 后端API: http://localhost:5000/api/health
echo.
echo 📊 主要功能页面：
echo   📈 负荷预测: http://localhost:5173/forecast
echo   🌤️  天气场景分析: http://localhost:5173/weather
echo   📊 数据可视化: http://localhost:5173/dashboard
echo.
echo 🛠️ 管理说明：
echo   - 两个窗口将自动打开（后端和前端）
echo   - 请保持两个窗口都运行
echo   - 按 Ctrl+C 在对应窗口中停止服务
echo   - 关闭此窗口不会影响服务运行
echo.
echo 🎯 如果遇到问题：
echo   1. 检查端口是否被占用
echo   2. 确认防火墙允许访问
echo   3. 查看各窗口的错误信息
echo.
echo 享受使用多源电力预测系统！
pause 