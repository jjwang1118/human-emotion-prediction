"""
訓練數據可視化工具
從 training_history.json 讀取數據並繪製圖表
"""
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YahHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.append('util')
from config_loader import load_config

def plot_training_history(config_path='config.yaml', history_path=None):
    """
    繪製訓練歷史曲線
    
    Args:
        config_path: 配置文件路徑
        history_path: training_history.json 的路徑（可選，從配置讀取）
    """
    # 載入配置
    config = load_config(config_path)
    
    # 如果沒有指定 history_path，從配置文件中讀取
    if history_path is None:
        history_path = config['paths']['training_history']
    
    if not os.path.exists(history_path):
        print(f"錯誤: 找不到訓練歷史文件 {history_path}")
        return
    
    # 載入訓練歷史
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 繪製損失曲線
    ax1.plot(epochs, train_loss, 'b-', label='訓練損失', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='驗證損失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('損失 (Loss)', fontsize=12)
    ax1.set_title('訓練與驗證損失', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 繪製準確率曲線
    ax2.plot(epochs, train_acc, 'b-', label='訓練準確率', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='驗證準確率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('準確率 (%)', fontsize=12)
    ax2.set_title('訓練與驗證準確率', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖表
    save_path = 'model/training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'✅ 圖表已保存至: {save_path}')
    
    # 顯示圖表
    plt.show()
    
    # 打印統計信息
    print(f'\n📊 訓練統計:')
    print(f'總 Epoch 數: {len(epochs)}')
    print(f'最佳驗證準確率: {max(val_acc):.2f}% (Epoch {epochs[val_acc.index(max(val_acc))]}, )')
    print(f'最終訓練準確率: {train_acc[-1]:.2f}%')
    print(f'最終驗證準確率: {val_acc[-1]:.2f}%')
    print(f'最低驗證損失: {min(val_loss):.4f} (Epoch {epochs[val_loss.index(min(val_loss))]})')

if __name__ == '__main__':
    plot_training_history()
