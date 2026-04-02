import torch 
from util.download_dataset import EmotionDataset
from util.load_create_model import model_create, load_model
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":

    test_set_path=  "datasets/fer2013"
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set=EmotionDataset(test_set_path, split="test", transform=test_transform)

    data=DataLoader(
        #   因為測試集不需要數據增強，所以直接傳入原始數據集（不使用 transform）
        #   因為是預測，所以不batch_size=1，確保每次只處理一張圖片
        test_set,
        shuffle=False,
        batch_size=1
    )

    model=load_model("model/best_model.pth", device="cuda")

    # 開始預測
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for idx, (img,label) in enumerate(data):
            img = img.to("cuda")
            output = model(img)
            predicted_label = torch.argmax(output, dim=1).item()
            print(f"Image {idx+1}: Predicted Label = {predicted_label}, True Label = {torch.argmax(label).item()}")
            if predicted_label == torch.argmax(label).item():
                correct_predictions += 1
            total_predictions += 1
    accuracy = correct_predictions / total_predictions * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
