import torch
import torch.nn as nn
import torch.nn.functional as F


def model_create(device="cuda"):
    """
    創建模型
    
    Args:
        device: 使用的設備 (cuda 或 cpu)
        
    Returns:
        model: 創建的模型
    """

    class model(torch.nn.Module):
        def __init__(self):
            super(model, self).__init__()
            
            # 左側分支 - Branch 1
            self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.bn1_1 = nn.BatchNorm2d(64)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn1_2 = nn.BatchNorm2d(64)
            self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn1_3 = nn.BatchNorm2d(128)
            self.conv1_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn1_4 = nn.BatchNorm2d(128)
            
            self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn2_1 = nn.BatchNorm2d(256)
            self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn2_2 = nn.BatchNorm2d(256)
            self.conv2_3= nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.bn2_3 = nn.BatchNorm2d(512)
            self.conv2_4= nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.bn2_4 = nn.BatchNorm2d(512)

            # 全連接層
            self.fc1 = nn.Linear(512 * 3 * 3, 4096)
            self.dropout1 = nn.Dropout(0.5)  # 增加到 0.5
            self.fc2 = nn.Linear(4096, 4096)
            self.dropout2 = nn.Dropout(0.5)  # 增加到 0.5
            self.fc3 = nn.Linear(4096, 7)  
            
        
        def forward(self, x):
            # 第1組卷積 - 64通道
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = F.relu(x)
            
            x = self.conv1_2(x)
            x = self.bn1_2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)  # 48x48 -> 24x24
            
            # 第2組卷積 - 128通道
            x = self.conv1_3(x)
            x = self.bn1_3(x)
            x = F.relu(x)
            
            x = self.conv1_4(x)
            x = self.bn1_4(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)  # 24x24 -> 12x12
            
            # 第3組卷積 - 256通道
            x = self.conv2_1(x)
            x = self.bn2_1(x)
            x = F.relu(x)
            
            x = self.conv2_2(x)
            x = self.bn2_2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)  # 12x12 -> 6x6
            
            # 第4組卷積 - 512通道
            x = self.conv2_3(x)
            x = self.bn2_3(x)
            x = F.relu(x)
            
            x = self.conv2_4(x)
            x = self.bn2_4(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)  # 6x6 -> 3x3
            
            # Flatten 展平 (512 * 3 * 3 = 4608)
            x = x.view(x.size(0), -1)
            
            # 全連接層 - Dense
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout1(x)
            
            x = self.fc2(x)
            x = F.relu(x)
            x = self.dropout2(x)
            
            x = self.fc3(x)  # 輸出7個類別
            
            return x
    model=model().to("cuda")
    return model


def load_model(checkpoint_path, device="cuda"):

    model = model_create()
    
    if checkpoint_path.endswith('final_model.pth'):
        # 只包含 state_dict
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        # 完整檢查點
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"載入模型 - Epoch: {checkpoint.get('epoch', 'N/A')}, "
              f"準確率: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    
    model.eval()
    return model
