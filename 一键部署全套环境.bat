@echo off
chcp 65001
echo ==========================================
echo 多源电力预测系统 - 全套环境一键部署
echo ==========================================

echo.
echo 🎯 此脚本将自动安装和配置：
echo   ✅ Python 3.9 环境
echo   ✅ Node.js LTS 环境
echo   ✅ 项目依赖包
echo   ✅ 启动完整系统
echo.

echo ⚠️  重要提示：
echo   - 需要管理员权限运行
echo   - 需要稳定的网络连接
echo   - 整个过程可能需要10-20分钟
echo   - 请耐心等待，不要中途关闭
echo.

echo 按任意键开始部署，或按Ctrl+C取消...
pause

echo.
echo ==========================================
echo 📦 第一阶段：环境检查与安装
echo ==========================================

echo.
echo [1/6] 检查管理员权限...
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ 需要管理员权限！
    echo 请右键点击此脚本，选择"以管理员身份运行"
    pause
    exit /b 1
)
echo ✅ 管理员权限检查通过

echo.
echo [2/6] 检查并安装Python...
python --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo 📦 正在安装Python 3.9...
    winget install Python.Python.3.9 --accept-package-agreements --accept-source-agreements
    if %ERRORLEVEL% neq 0 (
        echo ❌ Python安装失败！请手动安装或检查网络连接
        pause
        exit /b 1
    )
    echo ✅ Python安装成功
) else (
    echo ✅ Python已安装
    python --version
)

echo.
echo [3/6] 检查并安装Node.js...
node --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo 📦 正在安装Node.js LTS...
    winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
    if %ERRORLEVEL% neq 0 (
        echo ❌ Node.js安装失败！请手动安装或检查网络连接
        pause
        exit /b 1
    )
    echo ✅ Node.js安装成功
) else (
    echo ✅ Node.js已安装
    node --version
)

echo.
echo ==========================================
echo 🔧 第二阶段：环境配置
echo ==========================================

echo.
echo [4/6] 配置Python虚拟环境...
if not exist "venv\" (
    echo 📦 创建Python虚拟环境...
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
echo 📦 激活虚拟环境并安装Python依赖...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if %ERRORLEVEL% neq 0 (
    echo ❌ Python依赖安装失败！
    pause
    exit /b 1
)
echo ✅ Python依赖安装完成

echo.
echo [5/6] 配置前端环境...
cd frontend
echo 📦 配置npm镜像源...
npm config set registry https://registry.npmmirror.com
echo 📦 安装前端依赖...
npm install
if %ERRORLEVEL% neq 0 (
    echo ❌ 前端依赖安装失败！
    cd ..
    pause
    exit /b 1
)
echo ✅ 前端依赖安装完成
cd ..

echo.
echo ==========================================
echo 🚀 第三阶段：启动系统
echo ==========================================

echo.
echo [6/6] 启动完整系统...
echo 📍 启动顺序：后端 → 前端

echo.
echo 🔧 启动后端服务...
start "多源电力预测系统-后端" cmd /k "call venv\Scripts\activate.bat && python app.py"

echo ⏳ 等待后端启动完成...
timeout /t 8 /nobreak >nul

echo.
echo 🌐 启动前端服务...
start "多源电力预测系统-前端" cmd /k "cd frontend && npm run dev"

echo.
echo ==========================================
echo 🎉 部署完成！系统已启动
echo ==========================================

echo.
echo 📱 访问地址：
echo   🌐 前端界面: http://localhost:5173
echo   ⚡ 后端API: http://localhost:5000/api/health
echo.

echo 📊 主要功能页面：
echo   📈 负荷预测界面
echo   🌤️  天气场景分析
echo   📊 数据可视化图表
echo   🔧 系统配置管理
echo.

echo 🛠️ 使用说明：
echo   - 已自动打开两个服务窗口
echo   - 请保持两个窗口持续运行
echo   - 访问前端界面开始使用系统
echo   - 按Ctrl+C可在对应窗口停止服务
echo.

echo 🎯 如果访问异常，请：
echo   1. 等待1-2分钟让服务完全启动
echo   2. 检查防火墙设置
echo   3. 查看服务窗口的错误信息
echo   4. 重新运行此脚本
echo.

echo 🌟 恭喜！您已成功部署多源电力预测系统！
echo    现在可以开始使用这个强大的电力预测平台了！
echo.

pause 