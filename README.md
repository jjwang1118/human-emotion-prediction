# 人類情緒識別專案

基於 PyTorch 的情緒識別深度學習專案，使用 FER2013 數據集訓練 CNN 模型識別 7 種情緒。

## 功能特點

- ✅ 使用 YAML 配置文件管理所有參數
- ✅ 支援數據增強和早停機制
- ✅ TensorBoard 可視化訓練過程
- ✅ 模型檢查點自動保存
- ✅ 完整的訓練、評估和可視化工具

## 環境需求

- Python 3.8+
- CUDA 支援的 GPU（推薦）
- 8GB+ GPU 顯存

## 快速開始

### 方式一：本地安装

#### 1. 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 conda
conda create -n emotion_recognition python=3.9
conda activate emotion_recognition
pip install -r requirements.txt
```

### 方式二：Docker 部署（推荐）

#### 1. 安装 Docker

- Windows/Mac: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Linux: `curl -fsSL https://get.docker.com | sh`

#### 2. 构建并运行

```bash
# 构建镜像
docker-compose build

# 启动训练
docker-compose up train

# 查看 TensorBoard
docker-compose up tensorboard
# 访问 http://localhost:6006

# 评估模型
docker-compose run --rm eval
```

详细的 Docker 部署指南请参考 [DOCKER_README.md](DOCKER_README.md)

---

## 本地部署步骤

如果不使用 Docker，请按以下步骤操作：

### 2. 準備數據集

將 FER2013 數據集放置在 `datasets/fer2013/` 目錄下，結構如下：

```
datasets/fer2013/
  ├── train/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── neutral/
  │   ├── sad/
  │   └── surprise/
  └── test/
      ├── angry/
      ├── disgust/
      ├── fear/
      ├── happy/
      ├── neutral/
      ├── sad/
      └── surprise/
```

### 3. 配置參數

編輯 `config.yaml` 文件來調整訓練參數：

```yaml
training:
  batch_size: 64
  epochs: 200
  learning_rate: 0.001
  # ... 更多參數
```

詳細配置說明請參考 [CONFIG_README.md](CONFIG_README.md)

### 4. 訓練模型

```bash
python train.py
```

訓練過程會：
- 自動切分驗證集（默認 20%）
- 保存最佳模型到 `model/best_model.pth`
- 記錄訓練歷史到 `model/training_history.json`
- 生成 TensorBoard 日誌

### 5. 監控訓練

使用 TensorBoard 實時監控訓練過程：

```bash
tensorboard --logdir=runs
```

然後在瀏覽器中打開 `http://localhost:6006`

### 6. 評估模型

在測試集上評估最佳模型：

```bash
python eval.py
```

### 7. 可視化結果

繪製訓練曲線：

```bash
python visualize.py
```

## 項目結構

```
.
├── config.yaml              # 配置文件
├── requirements.txt         # 依賴項
├── train.py                 # 訓練腳本
├── eval.py                  # 評估腳本
├── visualize.py             # 可視化腳本
├── process.py               # 數據處理
├── test_config.py           # 配置測試
├── util/                    # 工具模塊
│   ├── download_dataset.py  # 數據集載入
│   ├── load_create_model.py # 模型定義
│   └── config_loader.py     # 配置載入
├── datasets/                # 數據集目錄
├── model/                   # 模型保存目錄
└── runs/                    # TensorBoard 日誌
```

## 模型架構

- 4 組卷積層（64 → 128 → 256 → 512 通道）
- 批次正規化（Batch Normalization）
- 最大池化（Max Pooling）
- 3 層全連接層（4096 → 4096 → 7）
- Dropout（0.5）防止過擬合

## 配置說明

### 主要參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| batch_size | 64 | 批次大小 |
| epochs | 200 | 最大訓練輪數 |
| learning_rate | 0.001 | 學習率 |
| early_stopping.patience | 20 | 早停耐心值 |
| val_split | 0.2 | 驗證集比例 |

更多配置選項請參考 [CONFIG_README.md](CONFIG_README.md)

## 測試配置

驗證配置文件是否正確：

```bash
python test_config.py
```

## 常見問題

### CUDA 內存不足

降低 batch_size：
```yaml
training:
  batch_size: 32  # 或更小
```

### 訓練過慢

檢查是否使用 GPU：
```python
import torch
print(torch.cuda.is_available())  # 應該輸出 True
```

### 修改模型保存位置

編輯 `config.yaml`：
```yaml
paths:
  model_save_dir: "your/custom/path"
```

## 進階用法

### 從檢查點繼續訓練

修改 `train.py` 載入檢查點：
```python
checkpoint = torch.load('model/checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 自定義數據增強

編輯 `config.yaml` 中的 `data_augmentation` 部分

## 授權

本專案僅供學習和研究使用。

## 貢獻

歡迎提交 Issue 和 Pull Request！
