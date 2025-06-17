# 本地Python部署指南

## 🎯 为什么选择本地Python部署？

由于您遇到了Docker网络连接问题，本地Python部署是更快更简单的解决方案：
- ✅ 无需依赖网络拉取Docker镜像
- ✅ 部署速度更快
- ✅ 调试更方便
- ✅ 资源占用更少

## 📋 部署步骤

### 步骤1：安装Python 3.9

**方法一：官方安装包（推荐）**
1. 访问Python官网：https://www.python.org/downloads/
2. 下载Python 3.9.19（或最新的3.9.x版本）
3. 运行安装程序时务必勾选：
   - ✅ "Add Python to PATH"
   - ✅ "Install pip"

**方法二：使用Winget（Windows 11推荐）**
```powershell
# 在管理员PowerShell中运行
winget install Python.Python.3.9
```

**方法三：使用Chocolatey**
```powershell
# 先安装Chocolatey，再安装Python
choco install python39
```

### 步骤2：验证Python安装

```powershell
# 重新打开PowerShell，验证安装
python --version
pip --version
```

应该看到类似输出：
```
Python 3.9.19
pip 24.0
```

### 步骤3：创建虚拟环境

```powershell
# 在项目目录中创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 你应该看到命令提示符前面有(venv)
```

### 步骤4：安装项目依赖

```powershell
# 确保在虚拟环境中
pip install --upgrade pip

# 安装项目依赖（使用国内镜像源加速）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 步骤5：启动应用

```powershell
# 启动Flask应用
python app.py
```

## 🛠 一键部署脚本

我为您创建了一个自动化部署脚本，只需双击运行即可！

**run_local.bat** - 一键本地部署脚本 