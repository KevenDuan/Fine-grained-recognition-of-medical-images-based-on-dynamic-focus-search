import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ChannelAttentionModule(nn.Module):
    """
    通道注意力模块
    关注不同通道的重要性
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层用于通道特征压缩和重构
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 输入形状: (Batch, C, H, W)
        batch_size, channels, _, _ = x.size()
        
        # 全局平均池化: (Batch, C, 1, 1) -> (Batch, C)
        avg_out = self.avg_pool(x).view(batch_size, channels)
        
        # 通道注意力权重: (Batch, C)
        channel_weights = self.fc(avg_out)
        
        # 重塑为: (Batch, C, 1, 1)
        channel_weights = channel_weights.view(batch_size, channels, 1, 1)
        
        # 应用注意力权重
        out = x * channel_weights
        
        return out, channel_weights

class ChannelAttentionNet(nn.Module):
    """
    使用通道注意力机制的网络
    消融实验模型：仅使用通道注意力
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(ChannelAttentionNet, self).__init__()
        
        # 特征提取主干 - ResNet-18
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
        
        # 通道注意力模块
        self.channel_attention = ChannelAttentionModule(in_channels=512)
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        前向传播
        输入 x: (Batch, 3, 224, 224) 的X光片
        """
        # 提取基础特征
        features = self.backbone(x)
        
        # 应用通道注意力
        focused_features, channel_weights = self.channel_attention(features)
        
        # 分类
        pooled = self.global_pool(focused_features)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        
        # 为了与其他模型保持一致的输出格式，生成一个空间掩码
        # 由于使用的是通道注意力，这里生成一个全1的掩码
        spatial_mask = torch.ones(x.size(0), 1, 224, 224, device=x.device)
        
        return logits, spatial_mask

# 测试模型
if __name__ == '__main__':
    print("正在实例化通道注意力网络...")
    model = ChannelAttentionNet(num_classes=3, pretrained=False)
    
    # 模拟输入
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("正在进行前向传播测试...")
    out_logits, out_mask = model(dummy_input)
    
    print(f"✅ 测试成功！")
    print(f"分类输出维度 (Logits): {out_logits.shape}")
    print(f"掩码输出维度 (Mask): {out_mask.shape}")