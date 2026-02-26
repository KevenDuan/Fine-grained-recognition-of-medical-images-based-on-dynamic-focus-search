"""
ResNet-18
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import COVIDFocusDataset 

if __name__ == '__main__':
    # 1. 加载数据
    dataset = COVIDFocusDataset(
        base_dir='./dataset', 
        split='Train', target_size=(224, 224)
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # 2. 硬件加速起飞！(自动识别 Mac 的 Apple Silicon 加速)
    device = torch.device("cpu")
    print(f"当前使用的计算设备: {device}")

    # 3. 召唤经典的 ResNet-18 预训练模型
    # 我们使用预训练权重 (weights='DEFAULT')，这样它一开始就自带了识别基础图形的能力
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 4. 魔改网络的最后一层 (分类头)
    # 原版 ResNet 是为了识别 ImageNet 的 1000 种物体设计的
    # 我们把最后的全连接层 (fc) 拆掉，换成只有 3 个输出节点的新层 (对应我们的 3 个类别)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    
    # 把模型搬到显存里
    model = model.to(device)

    # 5. 定义“监考老师” (损失函数)
    # 交叉熵损失函数，最适合做多分类任务
    criterion = nn.CrossEntropyLoss()

    # ================= 点火测试 =================
    print("正在进行网络前向传播测试...")
    
    # 从数据加载器里抓取一个批次的数据
    images, masks, labels = next(iter(dataloader))
    
    # 把数据也搬到对应的设备上
    images = images.to(device)
    labels = labels.to(device)

    # 让图片穿过神经网络，得到预测结果
    outputs = model(images)
    
    # 计算预测结果和标准答案之间的差距 (Loss)
    loss = criterion(outputs, labels)

    print(f"网络输出形状: {outputs.shape}") # 预期: [16, 3] (16张图，每张图3个类别的打分)
    print(f"当前批次的初始 Loss 值: {loss.item():.4f}")
    print("点火测试成功！模型和数据完美兼容！")