import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DynamicFocusNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(DynamicFocusNet, self).__init__()
        
        # ==========================================
        # 1. 特征提取主干 (Backbone) - 借用 ResNet-18 的眼睛
        # ==========================================
        # 我们加载预训练的 ResNet-18，但不要它最后的池化层和全连接层
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4  # 输出特征图大小为 (Batch, 512, 7, 7)
        )
        
        # ==========================================
        # 2. 核心创新：空间注意力生成模块 (Spatial Attention Module)
        # ==========================================
        # 它的任务是看着 512 通道的特征图，浓缩出一张 1 通道的“黑白探照灯”图
        self.attention_module = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 最后用 1x1 卷积压缩到 1 个通道，并用 Sigmoid 将所有值卡在 0~1 之间
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        
        # ==========================================
        # 3. 细粒度分类头 (Fine-grained Classification Head)
        # ==========================================
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        前向传播：定义图片在网络中流动的方向
        输入 x: (Batch, 3, 224, 224) 的 X 光片
        """
        # 第一步：全局扫视，提取基础特征
        # 提取后的 features 维度为 (Batch, 512, 7, 7)
        features = self.backbone(x)
        
        # 第二步：生成注意力权重图 (寻找病灶)
        # attn_mask 维度为 (Batch, 1, 7, 7)，里面的值全在 0 到 1 之间
        attn_mask = self.attention_module(features)
        
        # 🚨 第三步：强行聚焦 (动态搜索的核心数学体现！)
        # 把特征图和注意力图“相乘”。背景（接近0）被抹杀，病灶（接近1）被保留
        focused_features = features * attn_mask
        
        # 第四步：基于纯净的病灶特征，进行最终的确诊
        # 融合后的特征送入池化层和全连接层
        pooled = self.global_pool(focused_features)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        
        # 第五步：为了在训练时让医生掩码当“教鞭”，我们需要把 7x7 的注意力图放大回 224x224
        # 这样才能和你的 ground truth mask 计算误差！
        upsampled_attn_mask = F.interpolate(
            attn_mask, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 返回最终分类结果 和 放大后的模型注意力图
        return logits, upsampled_attn_mask

# 简单测试一下网络能不能跑通 (打桩测试)
if __name__ == '__main__':
    print("正在实例化动态聚焦网络...")
    model = DynamicFocusNet(num_classes=3, pretrained=False)
    
    # 模拟输入一张 224x224 的 RGB 图片 (Batch Size = 2)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("正在进行前向传播测试...")
    out_logits, out_mask = model(dummy_input)
    
    print(f"✅ 测试成功！")
    print(f"分类输出维度 (Logits): {out_logits.shape}  -> 期望是 (2, 3)")
    print(f"注意力图输出维度 (Mask): {out_mask.shape} -> 期望是 (2, 1, 224, 224)")