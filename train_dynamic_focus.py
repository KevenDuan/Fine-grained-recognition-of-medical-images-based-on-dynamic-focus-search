import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import COVIDFocusDataset 
from dynamic_focus_net import DynamicFocusNet  # 🚨 导入我们刚刚写好的端到端核心网络
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_dir = './dataset'

    print("正在加载训练集和验证集...")
    train_dataset = COVIDFocusDataset(base_dir=data_dir, split='Train', target_size=(224, 224))
    val_dataset = COVIDFocusDataset(base_dir=data_dir, split='Val', target_size=(224, 224))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"数据加载完成！当前计算设备: {device}")

    # ================= 🚨 核心改动 1：使用全新的端到端网络 =================
    model = DynamicFocusNet(num_classes=3, pretrained=True)
    model = model.to(device)

    # ================= 🚨 核心改动 2：定义双重损失函数 (联合监督) =================
    criterion_cls = nn.CrossEntropyLoss() # 用于分类的交叉熵损失
    criterion_mask = nn.MSELoss()         # 用于监督注意力图的均方误差损失
    
    # 这是一个极其重要的超参数，决定了“掩码教鞭”打得有多重。
    # 0.5 表示我们既看重分类，也看重聚焦定位。
    lambda_weight = 0.5 

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0.0
    
    # 记录数据的“小本本” (额外增加了对 Mask Loss 的记录，写论文必备！)
    history_train_loss_total = []
    history_train_loss_cls = []
    history_train_loss_mask = []
    
    history_val_loss_total = []
    history_val_acc = []
    
    num_epochs = 30
    for epoch in range(num_epochs):
        print(f"\n======== Epoch {epoch+1}/{num_epochs} ========")
        
        # ------------------ 训练阶段 ------------------
        model.train()
        running_loss_total = 0.0
        running_loss_cls = 0.0
        running_loss_mask = 0.0
        
        pbar_train = tqdm(train_loader, desc="[训练阶段]", unit="batch")
        # 注意：现在 dataset 吐出三个东西了！images, masks, labels
        for images, masks, labels in pbar_train:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # ================= 🚨 核心改动 3：接收两个输出，计算双重 Loss =================
            # 前向传播：同时得到分类结果 和 预测的注意力掩码
            outputs, pred_masks = model(images)
            
            # 1. 计算分类算错了多少
            loss_cls = criterion_cls(outputs, labels)
            # 2. 计算注意力图偏离了真实病灶多少
            loss_mask = criterion_mask(pred_masks, masks)
            
            # 3. 总误差 = 分类误差 + λ * 掩码误差 (这行代码值一篇核心论文！)
            loss_total = loss_cls + lambda_weight * loss_mask
            
            loss_total.backward()
            optimizer.step()
            
            running_loss_total += loss_total.item()
            running_loss_cls += loss_cls.item()
            running_loss_mask += loss_mask.item()
            
            pbar_train.set_postfix({
                'Total': f"{running_loss_total / (pbar_train.n + 1):.3f}",
                'Cls': f"{running_loss_cls / (pbar_train.n + 1):.3f}",
                'Mask': f"{running_loss_mask / (pbar_train.n + 1):.3f}"
            })

        avg_train_loss_total = running_loss_total / len(train_loader)
        history_train_loss_total.append(avg_train_loss_total)
        history_train_loss_cls.append(running_loss_cls / len(train_loader))
        history_train_loss_mask.append(running_loss_mask / len(train_loader))

        # ------------------ 验证阶段 ------------------
        model.eval() 
        val_loss_total = 0.0
        correct = 0   
        total = 0     
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="[验证阶段]", unit="batch")
            for images, masks, labels in pbar_val:
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                
                outputs, pred_masks = model(images)
                
                loss_cls = criterion_cls(outputs, labels)
                loss_mask = criterion_mask(pred_masks, masks)
                loss_total = loss_cls + lambda_weight * loss_mask
                
                val_loss_total += loss_total.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_acc = 100 * correct / total
        avg_val_loss_total = val_loss_total / len(val_loader)
        
        history_val_loss_total.append(avg_val_loss_total)
        history_val_acc.append(epoch_acc)
        
        print(f"👉 本轮成绩单: Train Loss: {avg_train_loss_total:.4f} | Val Loss: {avg_val_loss_total:.4f} | 准确率: {epoch_acc:.2f}%")
        
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), 'dynamic_focus_best_model.pth')
            print(f"🌟 新纪录！端到端模型已保存至 dynamic_focus_best_model.pth, 准确率: {best_val_acc:.2f}%")

    # ================= 绘制更加丰富的训练曲线图 =================
    print("\n训练结束，正在绘制多任务联合训练曲线图...")
    plt.figure(figsize=(18, 5)) # 加宽画布，画三张图
    
    # 1. 总 Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), history_train_loss_total, label='Train Total Loss')
    plt.plot(range(1, num_epochs + 1), history_val_loss_total, label='Val Total Loss')
    plt.title('Total Joint Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    
    # 2. 分类 Loss 与 掩码 Loss 分解对比 (论文核心亮点)
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), history_train_loss_cls, label='Train Cls Loss', linestyle='--')
    plt.plot(range(1, num_epochs + 1), history_train_loss_mask, label='Train Mask Loss', linestyle='-.')
    plt.title('Loss Breakdown (Cls vs Mask)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    
    # 3. 准确率曲线
    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs + 1), history_val_acc, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('joint_training_curve.png', dpi=300)
    print("✅ 多任务曲线图已成功保存为 joint_training_curve.png")