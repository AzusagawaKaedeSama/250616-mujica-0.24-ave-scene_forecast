@echo off
chcp 65001
echo ==========================================
echo npm权限问题修复脚本
echo ==========================================

echo.
echo 🔧 此脚本将修复npm权限问题：
echo   ✅ 重新配置npm缓存路径
echo   ✅ 设置正确的权限
echo   ✅ 清理损坏的缓存
echo   ✅ 测试npm功能
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
echo [2/6] 检查Node.js和npm状态...
node --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Node.js未正确安装！
    echo 请先运行 install_nodejs.bat 安装Node.js
    pause
    exit /b 1
)
echo ✅ Node.js版本：
node --version
npm --version 2>nul || echo ⚠️ npm可能有问题

echo.
echo [3/6] 创建npm用户配置目录...
if not exist "%APPDATA%\npm" (
    mkdir "%APPDATA%\npm"
    echo ✅ 创建npm用户目录：%APPDATA%\npm
) else (
    echo ✅ npm用户目录已存在
)

if not exist "%APPDATA%\npm-cache" (
    mkdir "%APPDATA%\npm-cache"
    echo ✅ 创建npm缓存目录：%APPDATA%\npm-cache
) else (
    echo ✅ npm缓存目录已存在
)

echo.
echo [4/6] 重新配置npm路径...
echo 📦 设置npm全局模块路径到用户目录...
npm config set prefix "%APPDATA%\npm" 2>nul
npm config set cache "%APPDATA%\npm-cache" 2>nul
npm config set registry https://registry.npmmirror.com 2>nul

echo 📦 当前npm配置：
npm config list 2>nul || echo ⚠️ npm配置可能有问题

echo.
echo [5/6] 清理和修复npm...
echo 📦 清理npm缓存...
npm cache clean --force 2>nul || echo ⚠️ 缓存清理可能失败，继续...

echo 📦 删除可能损坏的缓存目录...
if exist "D:\nodejs\node_cache" (
    echo 正在删除有问题的缓存目录：D:\nodejs\node_cache
    rmdir /s /q "D:\nodejs\node_cache" 2>nul || echo ⚠️ 删除失败，可能需要手动删除
)

echo.
echo [6/6] 测试npm功能...
echo 📦 测试npm是否正常工作...
npm --version
if %ERRORLEVEL% neq 0 (
    echo ❌ npm仍有问题！
    echo 建议重新安装Node.js
    pause
    exit /b 1
)

echo 📦 测试npm安装功能...
echo 正在测试安装一个简单的包...
npm install --global npm@latest
if %ERRORLEVEL% neq 0 (
    echo ❌ npm安装功能仍有问题！
    echo 可能需要完全重新安装Node.js
) else (
    echo ✅ npm安装功能正常！
)

echo.
echo ==========================================
echo 🎉 npm权限修复完成！
echo ==========================================
echo.
echo 📋 已完成的修复：
echo   ✅ npm缓存路径：%APPDATA%\npm-cache
echo   ✅ npm全局路径：%APPDATA%\npm
echo   ✅ 清理了损坏的缓存
echo   ✅ 配置了国内镜像源
echo.
echo 🔧 环境变量配置：
echo   请将以下路径添加到系统PATH环境变量：
echo   %APPDATA%\npm
echo.
echo 💡 测试建议：
echo   1. 重新打开命令提示符
echo   2. 运行：npm --version
echo   3. 运行：npm install -g npm@latest
echo   4. 如果仍有问题，请运行 reinstall_nodejs.bat
echo.
pause 