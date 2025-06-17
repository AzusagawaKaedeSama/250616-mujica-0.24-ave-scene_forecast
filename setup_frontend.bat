@echo off
chcp 65001
echo ==========================================
echo 前端环境配置和启动脚本
echo ==========================================

echo.
echo [1/6] 检查Node.js环境...
node --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Node.js未安装！
    echo 请先运行 install_nodejs.bat 安装Node.js
    pause
    exit /b 1
)

echo ✅ Node.js环境检查通过
node --version
npm --version

echo.
echo [2/6] 进入前端目录...
if not exist "frontend\" (
    echo ❌ frontend目录不存在！
    echo 请确保在项目根目录运行此脚本
    pause
    exit /b 1
)

cd frontend
echo ✅ 已进入frontend目录

echo.
echo [3/6] 配置npm镜像源和路径...
echo 📦 检查npm权限...
npm config set registry https://registry.npmmirror.com
npm config set cache "%APPDATA%\npm-cache"
npm config set prefix "%APPDATA%\npm"
echo ✅ npm配置已优化

echo.
echo [4/6] 安装前端依赖...
echo 正在安装依赖包，这可能需要几分钟时间...
npm install
if %ERRORLEVEL% neq 0 (
    echo ❌ 依赖安装失败！
    echo 尝试清除缓存后重新安装...
    npm cache clean --force
    npm install
    if %ERRORLEVEL% neq 0 (
        echo ❌ 重新安装仍然失败！
        echo 请检查网络连接或手动执行：
        echo   cd frontend
        echo   npm install
        pause
        exit /b 1
    )
)
echo ✅ 前端依赖安装完成

echo.
echo [5/6] 检查后端是否运行...
curl -s http://localhost:5000/api/health >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ⚠️  后端未运行！
    echo 请先在另一个命令窗口中运行后端：
    echo   run_local.bat
    echo.
    echo 按任意键继续启动前端（前端会等待后端启动）...
    pause
) else (
    echo ✅ 后端已运行，前端可以正常连接
)

echo.
echo [6/6] 启动前端开发服务器...
echo ==========================================
echo 🌐 正在启动前端服务器...
echo ==========================================
echo.
echo 前端启动后可通过以下地址访问：
echo   开发服务器: http://localhost:5173
echo   或: http://localhost:3000
echo.
echo 前端已配置代理，会自动转发API请求到后端(localhost:5000)
echo 按 Ctrl+C 可停止前端服务器
echo ==========================================

npm run dev 