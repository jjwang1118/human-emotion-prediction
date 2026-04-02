import kagglehub
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
def download_dataset(store_path : str,dataset_name:str):

    if os.path.exists(store_path + f"/{dataset_name}"):
        print("Dataset already exists, skipping download.")
    else:
        kagglehub.load_dataset(
            "msambare/fer2013",
            path=store_path + f"/{dataset_name}",
            quiet=True,
            adapter="pandas"
        )
    return store_path + f"/{dataset_name}"

def data_process(pixel_img):

    img_tensor = torch.tensor(pixel_img, dtype=torch.float32)
    
    # 如果是灰度圖 (H, W)，複製到3個通道
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  
    # 如果是RGB圖 (H, W, C)，轉換維度順序
    elif len(img_tensor.shape) == 3 and img_tensor.shape[2] == 3:
        img_tensor = img_tensor.permute(2, 0, 1) 
    
    # 正規化到 [0, 1]
    img_tensor = img_tensor / 255.0
    
    return img_tensor


    

def one_hot_encoding(emotion):
    emotion_dict = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6
    }
    one_hot_vector = torch.zeros(len(emotion_dict))
    one_hot_vector[emotion_dict[emotion]] = 1
    return one_hot_vector

class EmotionDataset(Dataset):
    def __init__(self, file_path, split="train", transform=None):
        """
        Args:
            file_path: 數據集根目錄（包含 train/ 和 test/ 文件夾）
            split: "train" 或 "test"
            store_path: 處理後數據的存儲路徑
            transform: torchvision transforms（數據增強）
        """
        self.data = []
        self.transform = transform
        split_path = os.path.join(file_path, split)
        
        # 獲取所有情緒類別目錄
        emotion_types = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        for emotion_type in emotion_types:
            emotion_path = os.path.join(split_path, emotion_type)            
            # 收集該情緒的所有圖片文件
            image_files = [f for f in os.listdir(emotion_path) if os.path.isfile(os.path.join(emotion_path, f))]
            
            for file in image_files:
                file_path_full = os.path.join(emotion_path, file)
                image_pixel = np.array(Image.open(file_path_full))
                self.data.append({
                    "pixel_img": image_pixel,
                    "emotion": emotion_type
                })

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        image = Image.fromarray(sample["pixel_img"])
        
        # 應用數據增強
        if self.transform:
            image = self.transform(image)
        else:
            image = data_process(sample["pixel_img"])
        
        label = one_hot_encoding(sample["emotion"])
        return image, label
    

