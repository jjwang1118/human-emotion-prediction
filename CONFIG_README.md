# 配置文件使用說明

本專案使用 YAML 配置文件來管理所有參數，使配置更加靈活和易於維護。

## 配置文件位置

`config.yaml` - 位於專案根目錄

## 配置結構

### 1. 模型配置 (model)
- `input_channels`: 輸入通道數
- `num_classes`: 分類類別數
- `dropout_rate`: Dropout 比例
- `device`: 訓練設備 (cuda/cpu)

### 2. 數據集配置 (dataset)
- `root_path`: 數據集根目錄
- `val_split`: 驗證集比例

### 3. 訓練配置 (training)
- `batch_size`: 批次大小
- `epochs`: 最大訓練輪數
- `learning_rate`: 學習率
- `weight_decay`: 權重衰減係數
- `early_stopping.patience`: 早停耐心值
- `scheduler`: 學習率調度器參數
- `checkpoint_interval`: 檢查點保存間隔

### 4. 數據增強配置 (data_augmentation)
- `train`: 訓練集數據增強參數

### 5. 路徑配置 (paths)
- `model_save_dir`: 模型保存目錄
- `best_model`: 最佳模型路徑
- `final_model`: 最終模型路徑
- `training_history`: 訓練歷史文件路徑
- `tensorboard_logs`: TensorBoard 日誌目錄

### 6. 評估配置 (evaluation)
- `batch_size`: 評估批次大小

## 使用方式

### 訓練模型
```bash
python train.py
```

訓練腳本會自動載入 `config.yaml` 中的參數。

### 評估模型
```bash
python eval.py
```

評估腳本會使用配置文件中指定的模型和數據集路徑。

### 可視化訓練結果
```bash
python visualize.py
```

可視化腳本會從配置文件中獲取訓練歷史文件路徑。

## 修改配置

直接編輯 `config.yaml` 文件即可修改參數，無需更改源代碼：

```yaml
# 例如：修改學習率
training:
  learning_rate: 0.0005  # 從 0.001 改為 0.0005
```

## 安裝依賴項

使用 requirements.txt 安裝所有依賴項：
```bash
pip install -r requirements.txt
```

或使用 conda 環境：
```bash
conda create -n emotion_recognition python=3.9
conda activate emotion_recognition
pip install -r requirements.txt
```

## 配置文件讀取

所有腳本通過 `util/config_loader.py` 中的 `load_config()` 函數讀取配置：

```python
from config_loader import load_config

config = load_config("config.yaml")
learning_rate = config['training']['learning_rate']
```
