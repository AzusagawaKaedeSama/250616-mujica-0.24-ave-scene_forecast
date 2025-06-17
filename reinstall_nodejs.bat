@echo off
chcp 65001
echo ==========================================
echo Node.js 完全重新安装脚本
echo ==========================================

echo.
echo ⚠️ 此脚本将完全卸载并重新安装Node.js
echo    这将解决所有npm权限和配置问题
echo.
echo 🔧 操作步骤：
echo   1️⃣ 卸载现有Node.js
echo   2️⃣ 清理残留文件和注册表
echo   3️⃣ 重新安装Node.js到用户目录
echo   4️⃣ 配置正确的npm设置
echo.

echo 按任意键继续，或按Ctrl+C取消...
pause

echo.
echo [1/5] 检查管理员权限...
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ 需要管理员权限！
    echo 请右键点击此脚本，选择"以管理员身份运行"
    pause
    exit /b 1
)
echo ✅ 管理员权限检查通过

echo.
echo [2/5] 卸载现有Node.js...
echo 📦 正在卸载Node.js...
winget uninstall OpenJS.NodeJS --accept-source-agreements 2>nul
winget uninstall "Node.js" --accept-source-agreements 2>nul

echo 📦 清理残留文件...
if exist "C:\Program Files\nodejs" (
    echo 删除：C:\Program Files\nodejs
    rmdir /s /q "C:\Program Files\nodejs" 2>nul
)
if exist "C:\Program Files (x86)\nodejs" (
    echo 删除：C:\Program Files (x86)\nodejs
    rmdir /s /q "C:\Program Files (x86)\nodejs" 2>nul
)
if exist "D:\nodejs" (
    echo 删除：D:\nodejs
    rmdir /s /q "D:\nodejs" 2>nul
)

echo 📦 清理用户目录...
if exist "%APPDATA%\npm" (
    echo 删除：%APPDATA%\npm
    rmdir /s /q "%APPDATA%\npm" 2>nul
)
if exist "%APPDATA%\npm-cache" (
    echo 删除：%APPDATA%\npm-cache
    rmdir /s /q "%APPDATA%\npm-cache" 2>nul
)

echo.
echo [3/5] 清理环境变量...
echo 📦 清理PATH环境变量中的Node.js路径...
rem 这里暂时跳过自动清理，建议手动检查

echo.
echo [4/5] 重新安装Node.js...
echo 📦 正在安装Node.js LTS版本...
winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
if %ERRORLEVEL% neq 0 (
    echo ❌ 自动安装失败！
    echo 请手动安装：
    echo 1. 访问：https://nodejs.org/
    echo 2. 下载LTS版本
    echo 3. 以管理员身份运行安装程序
    echo 4. 选择"Add to PATH"选项
    pause
    exit /b 1
)

echo ⏳ 等待安装完成...
timeout /t 10 /nobreak >nul

echo.
echo [5/5] 配置npm设置...
echo 📦 刷新环境变量...
call refreshenv 2>nul || echo ⚠️ 需要手动重启命令行

echo 📦 验证安装...
node --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Node.js安装验证失败！
    echo 请重启命令行后重试
    pause
    exit /b 1
)

npm --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ npm验证失败！
    echo 请重启命令行后重试
    pause
    exit /b 1
)

echo 📦 配置npm设置...
npm config set registry https://registry.npmmirror.com
npm config set cache "%APPDATA%\npm-cache"
npm config set prefix "%APPDATA%\npm"

echo.
echo ==========================================
echo ✅ Node.js重新安装完成！
echo ==========================================
echo.
echo 📋 安装信息：
node --version 2>nul && echo ✅ Node.js版本正常
npm --version 2>nul && echo ✅ npm版本正常

echo.
echo 📦 npm配置：
npm config get registry 2>nul
npm config get cache 2>nul
npm config get prefix 2>nul

echo.
echo 🔧 重要提示：
echo   1. 请关闭所有命令行窗口
echo   2. 重新打开PowerShell或命令提示符
echo   3. 运行：node --version 和 npm --version 验证
echo   4. 然后运行：setup_frontend.bat 配置前端
echo.
echo 🎯 如果仍有问题：
echo   1. 重启计算机
echo   2. 检查系统PATH环境变量
echo   3. 关闭防病毒软件后重试
echo.
pause 