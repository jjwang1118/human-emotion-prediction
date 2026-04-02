import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
sys.path.append('util')
from download_dataset import EmotionDataset

from train import model_create
from config_loader import load_config

def evaluate(set_name: str, config_path="config.yaml"):
    """
    評估模型在指定數據集上的準確率
    
    Args:
        set_name: "train" 或 "test"
        config_path: 配置文件路徑
    """
    # 載入配置
    config = load_config(config_path)
    
    # 定義 transform（測試時不做數據增強）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 載入數據集
    test_set = EmotionDataset(
        file_path=config['dataset']['root_path'],
        split=set_name,
        transform=test_transform
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
    )

    # 載入模型
    model = model_create()
    checkpoint = torch.load(config['paths']['best_model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 評估
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度計算
        for images, labels in test_loader:
            # 移到 GPU
            images = images.to("cuda")
            labels = labels.to("cuda")
            
            # 將 one-hot 轉換為類別索引
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)
            
            # 前向傳播
            outputs = model(images)
            
            # 獲取預測結果
            _, predicted = torch.max(outputs.data, 1)
            
            # 統計正確數量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\n在 {set_name} 集上的準確率: {accuracy:.2f}%")
    print(f"正確: {correct}/{total}")
    
    return accuracy

if __name__ == "__main__":
    print("=" * 50)
    print("模型評估")
    print("=" * 50)
    
    # 評估測試集
    test_acc = evaluate("test")
    
    # 也可以評估訓練集（用於檢查過擬合）
    # train_acc = evaluate("train")
    

