# 使用 PyTorch 官方 GPU 镜像作为基础
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p /app/model /app/runs /app/datasets

# 设置权限
RUN chmod -R 755 /app

# 暴露 TensorBoard 端口
EXPOSE 6006

# 默认命令
CMD ["python", "train.py"]
