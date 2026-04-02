# Docker 部署指南

本文档说明如何使用 Docker 和 Docker Compose 部署人类情绪识别项目。

## 前置要求

### 必需安装

1. **Docker** (版本 20.10+)
   - Windows: [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - Linux: `curl -fsSL https://get.docker.com | sh`

2. **Docker Compose** (通常随 Docker Desktop 一起安装)
   - 验证安装: `docker-compose --version`

3. **NVIDIA Docker 支持** (仅 GPU 训练需要)
   - 安装 NVIDIA 驱动
   - 安装 NVIDIA Container Toolkit

### GPU 支持设置 (Linux)

```bash
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证 GPU 可用
docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi
```

### GPU 支持设置 (Windows + WSL2)

1. 安装 [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install)
2. 安装 [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
3. 确保 Docker Desktop 启用 WSL2 后端

---

## 快速开始

### 1. 准备数据集

确保数据集在正确位置：
```
datasets/fer2013/
  ├── train/
  └── test/
```

### 2. 构建 Docker 镜像

```bash
# 构建镜像（首次运行或修改代码后需要）
docker-compose build

# 或使用缓存加速
docker-compose build --parallel
```

### 3. 训练模型

```bash
# 启动训练（GPU）
docker-compose up train

# 后台运行
docker-compose up -d train

# 查看日志
docker-compose logs -f train
```

### 4. 监控训练进度

```bash
# 启动 TensorBoard
docker-compose up tensorboard

# 在浏览器中访问: http://localhost:6006
```

### 5. 评估模型

```bash
# 训练完成后评估
docker-compose run --rm eval
```

### 6. 可视化结果

```bash
# 生成训练曲线图
docker-compose run --rm visualize
```

---

## 常用命令

### 构建和运行

```bash
# 构建镜像
docker-compose build

# 启动特定服务
docker-compose up <service_name>

# 后台运行
docker-compose up -d <service_name>

# 运行一次性命令
docker-compose run --rm <service_name>

# 停止所有服务
docker-compose down

# 停止并删除卷
docker-compose down -v
```

### 查看和调试

```bash
# 查看运行中的容器
docker-compose ps

# 查看日志
docker-compose logs <service_name>

# 实时跟踪日志
docker-compose logs -f <service_name>

# 进入容器 shell
docker-compose exec <service_name> bash

# 查看容器资源使用
docker stats
```

### 清理

```bash
# 删除未使用的镜像
docker image prune

# 删除所有停止的容器
docker container prune

# 完全清理（谨慎使用）
docker system prune -a --volumes
```

---

## 服务说明

### train
训练模型服务
- **GPU**: 必需
- **挂载卷**: datasets, model, runs, config.yaml
- **输出**: 最佳模型保存到 `model/best_model.pth`

### eval
评估模型服务
- **GPU**: 可选（推荐）
- **挂载卷**: datasets, model, config.yaml
- **输出**: 准确率输出到控制台

### tensorboard
可视化训练进度
- **GPU**: 不需要
- **端口**: 6006
- **访问**: http://localhost:6006

### visualize
生成训练曲线图
- **GPU**: 不需要
- **挂载卷**: model, config.yaml
- **输出**: 图表文件

### test-config
测试配置文件
- **GPU**: 不需要
- **挂载卷**: config.yaml
- **用途**: 验证配置正确性

---

## 数据卷说明

Docker Compose 会自动挂载以下目录：

| 主机路径 | 容器路径 | 用途 |
|---------|---------|------|
| ./datasets | /app/datasets | 数据集（只读） |
| ./model | /app/model | 模型保存 |
| ./runs | /app/runs | TensorBoard 日志 |
| ./config.yaml | /app/config.yaml | 配置文件 |

**注意**: 所有在容器中保存到这些路径的文件都会持久化到主机。

---

## 配置修改

### 修改训练参数

编辑 `config.yaml` 文件，修改会立即生效（无需重建镜像）：

```yaml
training:
  batch_size: 32  # 减小以节省内存
  epochs: 100     # 减少训练轮数
  learning_rate: 0.0005
```

### 使用 CPU 训练

如果没有 GPU，修改 `docker-compose.yml`：

```yaml
services:
  train:
    # 注释掉 runtime 和 deploy 部分
    # runtime: nvidia
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
```

同时修改 `config.yaml`:
```yaml
model:
  device: "cpu"
```

---

## 故障排除

### 问题 1: GPU 不可用

**错误**: `RuntimeError: CUDA not available`

**解决方案**:
```bash
# 检查主机 GPU
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi

# 重启 Docker
sudo systemctl restart docker  # Linux
# 或重启 Docker Desktop (Windows/Mac)
```

### 问题 2: 内存不足

**错误**: `CUDA out of memory`

**解决方案**:
- 减小 `batch_size` 在 config.yaml
- 限制 GPU 使用: 设置 `CUDA_VISIBLE_DEVICES=0` 在 docker-compose.yml

### 问题 3: 权限错误

**错误**: `Permission denied`

**解决方案**:
```bash
# Linux - 修复文件权限
sudo chown -R $USER:$USER model/ runs/

# 添加用户到 docker 组
sudo usermod -aG docker $USER
newgrp docker
```

### 问题 4: 端口已被占用

**错误**: `port is already allocated`

**解决方案**:
```bash
# 修改 docker-compose.yml 中的端口
ports:
  - "6007:6006"  # 改为其他端口
```

### 问题 5: 构建失败

**错误**: 构建过程中网络错误

**解决方案**:
```bash
# 使用国内镜像
# 在 Dockerfile 中添加:
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 性能优化

### 加速镜像构建

1. **使用构建缓存**
```bash
docker-compose build --parallel
```

2. **多阶段构建** (可选)
创建更小的生产镜像

### 加速训练

1. **调整 DataLoader 工作进程**
在 train.py 中设置:
```python
DataLoader(..., num_workers=4, pin_memory=True)
```

2. **使用混合精度训练**
需要修改 train.py 支持 AMP

---

## 生产部署建议

### 使用特定版本标签

```yaml
services:
  train:
    image: emotion-recognition:1.0.0  # 而不是 latest
```

### 设置资源限制

```yaml
services:
  train:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
```

### 健康检查

```yaml
services:
  tensorboard:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6006"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 高级用法

### 自定义训练命令

```bash
# 从检查点继续训练
docker-compose run --rm train python train.py --resume checkpoint_epoch_50.pth

# 使用自定义配置
docker-compose run --rm -v ./custom_config.yaml:/app/config.yaml train python train.py
```

### 多 GPU 训练

修改 docker-compose.yml:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # 使用两个 GPU
```

### CI/CD 集成

```bash
# 自动化测试
docker-compose run --rm test-config

# 构建并推送到仓库
docker tag emotion-recognition:latest your-registry.com/emotion-recognition:1.0.0
docker push your-registry.com/emotion-recognition:1.0.0
```

---

## 参考资源

- [Docker 官方文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [NVIDIA Docker 文档](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker 镜像](https://hub.docker.com/r/pytorch/pytorch)

---

## 获取帮助

遇到问题？

1. 查看日志: `docker-compose logs -f <service_name>`
2. 检查容器状态: `docker-compose ps`
3. 验证配置: `docker-compose config`
4. 查看项目 README.md 和 CONFIG_README.md
