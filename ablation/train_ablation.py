import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import COVIDFocusDataset 
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath('..'))

# 导入消融实验模型
from baseline_resnet import BaselineResNet
from channel_attention_net import ChannelAttentionNet
from hybrid_attention_net import HybridAttentionNet

def train_model(model_name, num_epochs=30, batch_size=256):
    """
    训练指定的消融实验模型
    
    Args:
        model_name: 模型名称，可选值: 'baseline', 'channel_attention', 'hybrid_attention'
        num_epochs: 训练轮数
        batch_size: 批次大小
    """
    print(f"\n=========================================")
    print(f"开始训练 {model_name} 模型...")
    print(f"=========================================")
    
    # 1. 加载数据集
    data_dir = '../dataset'
    print("正在加载训练集和验证集...")
    train_dataset = COVIDFocusDataset(base_dir=data_dir, split='Train', target_size=(224, 224))
    val_dataset = COVIDFocusDataset(base_dir=data_dir, split='Val', target_size=(224, 224))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"数据加载完成！当前计算设备: {device}")
    
    # 2. 初始化模型
    if model_name == 'baseline':
        model = BaselineResNet(num_classes=3, pretrained=True)
    elif model_name == 'channel_attention':
        model = ChannelAttentionNet(num_classes=3, pretrained=True)
    elif model_name == 'hybrid_attention':
        model = HybridAttentionNet(num_classes=3, pretrained=True)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
    
    model = model.to(device)
    
    # 3. 定义损失函数和优化器
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mask = nn.MSELoss()
    lambda_weight = 0.5  # 掩码损失的权重
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 4. 训练记录
    best_val_acc = 0.0
    history_train_loss_total = []
    history_train_loss_cls = []
    history_train_loss_mask = []
    history_val_loss_total = []
    history_val_acc = []
    
    # 5. 开始训练
    for epoch in range(num_epochs):
        print(f"\n======== Epoch {epoch+1}/{num_epochs} ========")
        
        # 训练阶段
        model.train()
        running_loss_total = 0.0
        running_loss_cls = 0.0
        running_loss_mask = 0.0
        
        pbar_train = tqdm(train_loader, desc="[训练阶段]", unit="batch")
        for images, masks, labels in pbar_train:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs, pred_masks = model(images)
            
            # 计算损失
            loss_cls = criterion_cls(outputs, labels)
            loss_mask = criterion_mask(pred_masks, masks)
            loss_total = loss_cls + lambda_weight * loss_mask
            
            # 反向传播
            loss_total.backward()
            optimizer.step()
            
            # 记录损失
            running_loss_total += loss_total.item()
            running_loss_cls += loss_cls.item()
            running_loss_mask += loss_mask.item()
            
            pbar_train.set_postfix({
                'Total': f"{running_loss_total / (pbar_train.n + 1):.3f}",
                'Cls': f"{running_loss_cls / (pbar_train.n + 1):.3f}",
                'Mask': f"{running_loss_mask / (pbar_train.n + 1):.3f}"
            })

        # 记录训练损失
        avg_train_loss_total = running_loss_total / len(train_loader)
        history_train_loss_total.append(avg_train_loss_total)
        history_train_loss_cls.append(running_loss_cls / len(train_loader))
        history_train_loss_mask.append(running_loss_mask / len(train_loader))

        # 验证阶段
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
                
        # 计算验证指标
        epoch_acc = 100 * correct / total
        avg_val_loss_total = val_loss_total / len(val_loader)
        
        history_val_loss_total.append(avg_val_loss_total)
        history_val_acc.append(epoch_acc)
        
        print(f"👉 本轮成绩单: Train Loss: {avg_train_loss_total:.4f} | Val Loss: {avg_val_loss_total:.4f} | 准确率: {epoch_acc:.2f}%")
        
        # 保存最佳模型
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            os.makedirs('ablation_results', exist_ok=True)
            torch.save(model.state_dict(), f'ablation_results/{model_name}_best_model.pth')
            print(f"🌟 新纪录！模型已保存至 ablation_results/{model_name}_best_model.pth, 准确率: {best_val_acc:.2f}%")
    
    # 绘制训练曲线
    print("\n训练结束，正在绘制训练曲线图...")
    plt.figure(figsize=(18, 5))
    
    # 总 Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), history_train_loss_total, label='Train Total Loss')
    plt.plot(range(1, num_epochs + 1), history_val_loss_total, label='Val Total Loss')
    plt.title(f'{model_name} - Total Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    
    # 分类 Loss 与 掩码 Loss 分解对比
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), history_train_loss_cls, label='Train Cls Loss', linestyle='--')
    plt.plot(range(1, num_epochs + 1), history_train_loss_mask, label='Train Mask Loss', linestyle='-.')
    plt.title(f'{model_name} - Loss Breakdown')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs + 1), history_val_acc, label='Validation Accuracy', color='green')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('ablation_results', exist_ok=True)
    plt.savefig(f'ablation_results/{model_name}_training_curve.png', dpi=300)
    print(f"✅ 训练曲线图已成功保存为 ablation_results/{model_name}_training_curve.png")
    
    return best_val_acc

if __name__ == '__main__':
    # 训练所有消融实验模型
    models_to_train = ['baseline', 'channel_attention', 'hybrid_attention']
    
    results = {}
    for model_name in models_to_train:
        best_acc = train_model(model_name)
        results[model_name] = best_acc
    
    # 打印所有模型的结果
    print("\n=========================================")
    print("消融实验结果汇总")
    print("=========================================")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.2f}%")