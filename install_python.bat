@echo off
chcp 65001
echo ==========================================
echo Python 3.9 自动安装脚本
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
echo [2/3] 检查是否已安装Python...
python --version 2>nul
if %ERRORLEVEL% equ 0 (
    echo ✅ Python已安装：
    python --version
    echo.
    echo Python已存在，是否继续安装？(y/n)
    set /p continue=
    if /i not "%continue%"=="y" (
        echo 取消安装
        pause
        exit /b 0
    )
)

echo.
echo [3/3] 使用Winget安装Python 3.9...
echo 正在安装Python，请稍候...
winget install Python.Python.3.9 --accept-package-agreements --accept-source-agreements

if %ERRORLEVEL% equ 0 (
    echo ✅ Python安装成功！
    echo.
    echo 请关闭当前命令窗口，重新打开PowerShell或命令提示符
    echo 然后运行: python --version 验证安装
    echo.
    echo 安装完成后，请运行 run_local.bat 启动项目
) else (
    echo ❌ 自动安装失败！
    echo.
    echo 请手动安装Python：
    echo 1. 访问：https://www.python.org/downloads/
    echo 2. 下载Python 3.9.x版本
    echo 3. 运行安装程序
    echo 4. 勾选"Add Python to PATH"
    echo 5. 点击"Install Now"
)

echo.
pause 