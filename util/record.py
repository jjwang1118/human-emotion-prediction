import csv
import os
from datetime import datetime


def record_experiment(config: dict, history: dict, record_path: str = "record.csv") -> None:
    """
    將訓練實驗結果記錄至 CSV 檔案。

    Parameters
    ----------
    config      : 訓練配置字典（來自 config.yaml）
    history     : 訓練歷史字典，需含 train_loss / train_acc / val_loss / val_acc
    record_path : 輸出 CSV 的路徑（預設 record.csv）
    """
    exp_name = f"exp_{config['experiment']['name']}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mean_train_loss = sum(history["train_loss"]) / len(history["train_loss"])
    mean_train_acc  = sum(history["train_acc"])  / len(history["train_acc"])
    mean_val_loss   = sum(history["val_loss"])   / len(history["val_loss"])
    mean_val_acc    = sum(history["val_acc"])    / len(history["val_acc"])

    fieldnames = [
        "experiment",
        "timestamp",
        "mean_train_loss",
        "mean_train_acc",
        "mean_val_loss",
        "mean_val_acc",
    ]

    row = {
        "experiment":      exp_name,
        "timestamp":       timestamp,
        "mean_train_loss": round(mean_train_loss, 6),
        "mean_train_acc":  round(mean_train_acc,  4),
        "mean_val_loss":   round(mean_val_loss,   6),
        "mean_val_acc":    round(mean_val_acc,    4),
    }

    file_exists = os.path.isfile(record_path)
    with open(record_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"📋 實驗記錄已寫入: {record_path}  [{exp_name}  {timestamp}]")
