import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
from datetime import datetime
import sys
sys.path.append('util')
from download_dataset import EmotionDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from load_create_model import model_create, load_model
from config_loader import load_config
from record import record_experiment

if __name__ == "__main__":

    # 載入配置文件
    config = load_config("config.yaml")
    print("✅ 配置文件載入成功")
    
    # 創建保存目錄
    save_dir = config['paths']['model_save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化 TensorBoard
    log_dir = os.path.join(config['paths']['tensorboard_logs'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日誌目錄: {log_dir}")
    
    # 初始化模型和優化器
    model = model_create()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    # 添加學習率調度器（降低過擬合）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['training']['scheduler']['mode'],
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience']
    )
    
    # 定義數據增強（訓練集使用）
    aug_cfg = config['data_augmentation']['train']
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=aug_cfg['random_horizontal_flip']),
        transforms.RandomAffine(
            degrees=aug_cfg['random_affine']['degrees'],
            translate=tuple(aug_cfg['random_affine']['translate']),
            scale=tuple(aug_cfg['random_affine']['scale'])
        ),
        transforms.RandomRotation(aug_cfg['random_rotation']),
        transforms.ColorJitter(
            brightness=aug_cfg['color_jitter']['brightness'], 
            contrast=aug_cfg['color_jitter']['contrast']
        ),
        transforms.ToTensor(),
    ])
    
    # 驗證集只需要 ToTensor（不做數據增強）
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 載入訓練數據集（傳入 transform）
    train_dataset = EmotionDataset(
        file_path=config['dataset']['root_path'], 
        split="train",
        transform=train_transform
    )

    # 切分驗證集（用於調參和早停）
    val_size = int(config['dataset']['val_split'] * len(train_dataset) + 1)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # 載入測試數據集（用於最終評估）
    test_dataset = EmotionDataset(
        file_path=config['dataset']['root_path'], 
        split="test",
        transform=val_transform
    )
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 訓練歷史記錄
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": []
    }
    
    best_val_acc = 0.0
    patience = config['training']['early_stopping']['patience']
    patience_counter = 0
    
    max_epochs = config['training']['epochs']
    print(f"開始訓練... (最多 {max_epochs} 個 epoch)")
    for epoch in range(max_epochs):
        # ===== 訓練階段 =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to("cuda")
            label = label.to("cuda")
            
            if label.dim() > 1:
                label = torch.argmax(label, dim=1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() 
            _, predicted = torch.max(output.data, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{max_epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # ===== 驗證階段 =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, label in val_loader:
                data = data.to("cuda")
                label = label.to("cuda")
                
                if label.dim() > 1:
                    label = torch.argmax(label, dim=1)
                
                output = model(data)
                loss = criterion(output, label)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # 保存訓練歷史
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)
        history["epochs"].append(epoch + 1)
        
        # 寫入 TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch + 1)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch + 1)
        
        print(f'\n===== Epoch {epoch+1}/{max_epochs} 完成 =====')
        print(f'訓練損失: {avg_train_loss:.4f}, 訓練準確率: {train_accuracy:.2f}%')
        print(f'驗證損失: {avg_val_loss:.4f}, 驗證準確率: {val_accuracy:.2f}%\n')
        
        # 調整學習率
        scheduler.step(val_accuracy)
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0  # 重置計數器
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': avg_val_loss
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'✅ 保存最佳模型 (準確率: {val_accuracy:.2f}%)')
        else:
            patience_counter += 1
            print(f'⏳ 驗證準確率未改善 ({patience_counter}/{patience})')
            
            # 早停
            if patience_counter >= patience:
                print(f'\n🛑 早停觸發！已經 {patience} 個 epoch 沒有改善')
                print(f'最佳驗證準確率: {best_val_acc:.2f}%')
                break
        
        # 定期保存檢查點
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_accuracy': val_accuracy
            }, checkpoint_path)
            print(f'💾 保存檢查點: {checkpoint_path}')
    
    # 保存最終模型
    final_model_path = config['paths']['final_model']
    torch.save(model.state_dict(), final_model_path)
    
    # 保存訓練歷史為 JSON
    history_path = config['paths']['training_history']
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    # 記錄實驗結果至 record.csv
    record_experiment(config, history)

    # 關閉 TensorBoard writer
    writer.close()
    
    # ===== 測試集評估 =====
    print(f'\n📊 開始在測試集上評估最佳模型...')
    
    # 載入最佳模型
    best_model = model_create()
    checkpoint = torch.load(config['paths']['best_model'])
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to("cuda")
            label = label.to("cuda")
            
            if label.dim() > 1:
                label = torch.argmax(label, dim=1)
            
            output = best_model(data)
            loss = criterion(output, label)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    
    print(f'\n{"="*50}')
    print(f'📈 最終測試集結果')
    print(f'{"="*50}')
    print(f'測試損失: {avg_test_loss:.4f}')
    print(f'測試準確率: {test_accuracy:.2f}%')
    print(f'{"="*50}\n')
    
    print(f'\n✅ 訓練完成！')
    print(f'最佳驗證準確率: {best_val_acc:.2f}%')
    print(f'最終測試準確率: {test_accuracy:.2f}%')
    print(f'最終模型保存至: {final_model_path}')
    print(f'訓練歷史保存至: {history_path}')
    print(f'TensorBoard 日誌目錄: {log_dir}')



