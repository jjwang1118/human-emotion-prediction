# Makefile for Docker operations
# 使用: make <target>

# 变量
IMAGE_NAME = emotion-recognition
CONTAINER_NAME = emotion-train

.PHONY: help build train eval tensorboard visualize test-config clean logs shell

# 默认目标：显示帮助
help:
	@echo "可用命令:"
	@echo "  make build        - 构建 Docker 镜像"
	@echo "  make train        - 启动训练"
	@echo "  make train-bg     - 后台启动训练"
	@echo "  make eval         - 评估模型"
	@echo "  make tensorboard  - 启动 TensorBoard"
	@echo "  make visualize    - 生成可视化图表"
	@echo "  make test-config  - 测试配置文件"
	@echo "  make logs         - 查看训练日志"
	@echo "  make shell        - 进入容器 shell"
	@echo "  make stop         - 停止所有容器"
	@echo "  make clean        - 清理容器和镜像"
	@echo "  make status       - 查看容器状态"

# 构建镜像
build:
	docker-compose build

# 快速构建（使用缓存）
build-fast:
	docker-compose build --parallel

# 训练模型
train:
	docker-compose up train

# 后台训练
train-bg:
	docker-compose up -d train
	@echo "训练已在后台启动"
	@echo "使用 'make logs' 查看日志"

# 评估模型
eval:
	docker-compose run --rm eval

# 启动 TensorBoard
tensorboard:
	@echo "启动 TensorBoard..."
	@echo "访问: http://localhost:6006"
	docker-compose up tensorboard

# 后台启动 TensorBoard
tensorboard-bg:
	docker-compose up -d tensorboard
	@echo "TensorBoard 已启动"
	@echo "访问: http://localhost:6006"

# 可视化
visualize:
	docker-compose run --rm visualize

# 测试配置
test-config:
	docker-compose run --rm test-config

# 查看日志
logs:
	docker-compose logs -f train

# 查看所有日志
logs-all:
	docker-compose logs -f

# 进入容器 shell
shell:
	docker-compose run --rm train bash

# 查看容器状态
status:
	docker-compose ps

# 停止所有容器
stop:
	docker-compose down

# 清理容器
clean-containers:
	docker-compose down -v

# 清理镜像
clean-images:
	docker rmi $(IMAGE_NAME):latest || true

# 完全清理
clean: clean-containers clean-images
	@echo "清理完成"

# 重新构建
rebuild: clean-containers build

# 验证 GPU
check-gpu:
	docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi

# 查看资源使用
stats:
	docker stats

# 一键运行（构建 + 训练 + TensorBoard）
run-all: build train-bg tensorboard-bg
	@echo "全部服务已启动"
	@echo "训练: 后台运行 (make logs 查看)"
	@echo "TensorBoard: http://localhost:6006"
