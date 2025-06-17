# Docker网络问题解决指南

## 🔧 您遇到的问题

您在新的Windows电脑上部署负荷预测项目时遇到了经典的Docker网络连接问题：

```
failed to authorize: failed to fetch oauth token: Post "https://auth.docker.io/token": 
read tcp 192.168.168.193:58099->98.85.153.80:443: wsarecv: 
An existing connection was forcibly closed by the remote host.
```

这个问题通常是由于**网络访问限制**或**Docker Hub连接不稳定**造成的。

## 🚀 解决方案（按优先级执行）

### 方案一：配置Docker镜像加速器（最推荐）

1. **打开Docker Desktop**
2. **点击设置（齿轮图标）**
3. **选择"Docker Engine"**
4. **在JSON配置中添加以下内容**：

```json
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://dockerproxy.com",
    "https://mirror.baidubce.com"
  ],
  "dns": ["8.8.8.8", "114.114.114.114"],
  "insecure-registries": [],
  "experimental": false
}
```

5. **点击"Apply & Restart"重启Docker**

### 方案二：使用优化的部署脚本

**运行新的优化部署脚本**：
```cmd
# 双击运行或在PowerShell中执行
deploy_fix.bat
```

这个脚本会：
- 自动检查Docker环境
- 清理旧的容器和镜像
- 预拉取镜像
- 提供详细的错误诊断

### 方案三：手动分步部署（如果上述方法失败）

```powershell
# 1. 停止现有容器
docker-compose down

# 2. 清理Docker缓存
docker system prune -f

# 3. 手动拉取镜像
docker pull python:3.9-slim

# 4. 重新构建
docker-compose build --no-cache

# 5. 启动服务
docker-compose up -d
```

### 方案四：网络诊断和修复

```powershell
# 1. 检查网络连接
ping 8.8.8.8

# 2. 检查DNS解析
nslookup docker.io

# 3. 重置Docker网络
docker network prune
docker network create --driver bridge scene_forecast_network

# 4. 重启Docker Desktop
# 右键Docker Desktop图标 -> Restart
```

## 🛠 文件修改说明

我已经为您优化了以下文件：

1. **docker-compose.yml** - 移除了过时的version字段
2. **Dockerfile** - 添加了国内镜像源加速
3. **deploy_fix.bat** - 新的优化部署脚本

## 📝 快速部署步骤

1. **配置Docker镜像加速器**（按方案一操作）
2. **重启Docker Desktop**
3. **运行优化部署脚本**：
   ```cmd
   deploy_fix.bat
   ```
4. **验证部署**：
   - 浏览器访问：http://localhost:5000/api/health
   - 如果看到健康检查响应，说明部署成功！

## 🚨 常见问题排查

### Q1: 仍然无法拉取镜像
**解决方法**：
- 检查公司网络是否有防火墙限制
- 尝试使用手机热点网络
- 联系网络管理员开放Docker相关端口

### Q2: 容器启动后立即退出
**检查命令**：
```powershell
docker-compose logs scene-forecast
```

### Q3: 端口被占用
**解决方法**：
```powershell
# 查看端口占用
netstat -ano | findstr :5000

# 修改端口（在docker-compose.yml中）
ports:
  - "5001:5000"  # 改为5001端口
```

## 🎯 项目部署成功后的访问地址

- **主应用**：http://localhost:5000
- **健康检查**：http://localhost:5000/api/health
- **API文档**：http://localhost:5000/api/

## 📞 如果问题仍未解决

请提供以下信息：
1. 运行 `docker-compose logs` 的完整输出
2. 您的网络环境（公司网络/家庭网络/移动热点）
3. 是否使用了VPN或代理

---

**温馨提示**：Docker网络问题很常见，特别是在新的Windows环境中。按照上述步骤操作，99%的情况都能解决！ 