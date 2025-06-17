@echo off
chcp 65001 > nul
echo.
echo ========================================
echo    HTML转PDF快速启动工具
echo ========================================
echo.
echo 正在用Chrome浏览器打开技术演示网页...
echo.

REM 获取当前脚本目录
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "HTML_FILE=%PROJECT_DIR%\docs\demo_webpage.html"

REM 检查文件是否存在
if not exist "%HTML_FILE%" (
    echo ❌ 错误：找不到文件 %HTML_FILE%
    echo.
    pause
    exit /b 1
)

echo 📂 文件位置: %HTML_FILE%
echo.

REM 尝试用Chrome打开
echo 🌐 正在启动Chrome浏览器...

REM 多种Chrome路径
set "CHROME1=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
set "CHROME2=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
set "CHROME3=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"

if exist "%CHROME1%" (
    start "" "%CHROME1%" "%HTML_FILE%"
    goto :opened
)

if exist "%CHROME2%" (
    start "" "%CHROME2%" "%HTML_FILE%"
    goto :opened
)

if exist "%CHROME3%" (
    start "" "%CHROME3%" "%HTML_FILE%"
    goto :opened
)

REM 如果找不到Chrome，用默认浏览器打开
echo ⚠️  未找到Chrome，使用默认浏览器...
start "" "%HTML_FILE%"

:opened
echo.
echo ✅ 网页已打开！
echo.
echo 📋 转换PDF操作步骤：
echo    1. 等待网页完全加载
echo    2. 按 Ctrl+P 打开打印对话框
echo    3. 选择 "保存为PDF"
echo    4. 在 "更多设置" 中启用 "背景图形"
echo    5. 点击 "保存" 并选择保存位置
echo.
echo 💡 建议保存文件名：
echo    基于多源数据融合的电网概率负荷预测方法_技术演示.pdf
echo.
echo 📖 详细指南请查看：docs\HTML转PDF指南.md
echo.
pause 