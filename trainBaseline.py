import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import COVIDFocusDataset 
from tqdm import tqdm
import matplotlib.pyplot as plt  # 新增：导入画图库

if __name__ == '__main__':
    data_dir = './dataset'

    print("正在加载训练集和验证集...")
    train_dataset = COVIDFocusDataset(base_dir=data_dir, split='Train', target_size=(224, 224))
    val_dataset = COVIDFocusDataset(base_dir=data_dir, split='Val', target_size=(224, 224))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"数据加载完成！当前计算设备: {device}")

    # 迁移学习：使用预训练的 ResNet-18 模型
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0.0
    
    # ================= 新增：准备记录数据的“小本本” =================
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []
    
    num_epochs = 30
    for epoch in range(num_epochs):
        print(f"\n======== Epoch {epoch+1}/{num_epochs} ========")
        
        # ------------------ 训练阶段 ------------------
        model.train()
        running_loss = 0.0
        
        pbar_train = tqdm(train_loader, desc="[训练阶段]", unit="batch")
        for images, masks, labels in pbar_train:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar_train.set_postfix({'Train Loss': f"{running_loss / (pbar_train.n + 1):.4f}"})

        # 计算本轮的平均训练误差，并记入小本本
        avg_train_loss = running_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # ------------------ 验证阶段 ------------------
        model.eval() 
        val_loss = 0.0
        correct = 0   
        total = 0     
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="[验证阶段]", unit="batch")
            for images, masks, labels in pbar_val:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # 计算本轮的验证误差和准确率，并记入小本本
        epoch_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(epoch_acc)
        
        print(f"👉 本轮成绩单: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | 准确率: {epoch_acc:.2f}%")
        
        # 保存最佳模型
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), 'resnet18_best_model.pth')
            print(f"🌟 模型已保存至 resnet18_best_model.pth, 准确率: {best_val_acc:.2f}%")

    # ================= 新增：全部跑完后，绘制并保存训练曲线图 =================
    print("\n训练结束，正在绘制训练曲线图...")
    
    # 设置画布大小，1行2列的并排图
    plt.figure(figsize=(12, 5))
    
    # 画第一张图：Loss 曲线 (对比 Train 和 Val)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history_train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), history_val_loss, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 画第二张图：Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history_val_acc, label='Validation Accuracy', color='green', marker='s')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300) # dpi=300 保证图片放在论文里足够高清
    print("✅ 曲线图已成功保存为 training_curve.png")