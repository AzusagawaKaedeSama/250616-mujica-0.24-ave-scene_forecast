@echo off
chcp 65001
echo ==========================================
echo Node.js 自动安装脚本
echo ==========================================

echo.
echo [1/3] 检查管理员权限...
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ 需要管理员权限！
    echo 请右键点击此脚本，选择"以管理员身份运行"
    pause
    exit /b 1
)
echo ✅ 管理员权限检查通过

echo.
echo [2/3] 检查Node.js安装状态...
node --version 2>nul
if %ERRORLEVEL% equ 0 (
    echo ✅ Node.js已安装：
    node --version
    npm --version
    echo.
    echo Node.js已存在，是否重新安装？(y/n)
    set /p continue=
    if /i not "%continue%"=="y" (
        echo 取消安装
        pause
        exit /b 0
    )
)

echo.
echo [3/3] 使用Winget安装Node.js LTS版本...
echo 正在安装Node.js，请稍候...
winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements

echo.
if %ERRORLEVEL% equ 0 (
    echo ✅ Node.js安装成功！
    echo.
    echo 请关闭当前命令窗口，重新打开PowerShell
    echo 然后运行以下命令验证安装：
    echo   node --version
    echo   npm --version
    echo.
    echo 验证成功后，请运行 setup_frontend.bat 配置前端
) else (
    echo ❌ 自动安装失败！
    echo.
    echo 请手动安装Node.js：
    echo 1. 访问：https://nodejs.org/
    echo 2. 下载LTS版本（推荐）
    echo 3. 运行安装程序，保持默认设置
    echo 4. 安装完成后重启命令提示符
)

echo.
pause 