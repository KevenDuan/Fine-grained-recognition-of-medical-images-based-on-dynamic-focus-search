import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ChannelAttentionModule(nn.Module):
    """
    通道注意力模块
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        avg_out = self.avg_pool(x).view(batch_size, channels)
        channel_weights = self.fc(avg_out)
        channel_weights = channel_weights.view(batch_size, channels, 1, 1)
        return x * channel_weights

class SpatialAttentionModule(nn.Module):
    """
    空间注意力模块
    """
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        spatial_weights = self.conv(x)
        return x * spatial_weights, spatial_weights

class HybridAttentionNet(nn.Module):
    """
    使用混合注意力机制的网络
    消融实验模型：同时使用通道注意力和空间注意力
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(HybridAttentionNet, self).__init__()
        
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
        
        # 混合注意力模块
        self.channel_attention = ChannelAttentionModule(in_channels=512)
        self.spatial_attention = SpatialAttentionModule(in_channels=512)
        
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
        channel_attended = self.channel_attention(features)
        
        # 应用空间注意力
        focused_features, spatial_mask = self.spatial_attention(channel_attended)
        
        # 分类
        pooled = self.global_pool(focused_features)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        
        # 上采样空间掩码到原始图像大小
        upsampled_mask = F.interpolate(
            spatial_mask, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        return logits, upsampled_mask

# 测试模型
if __name__ == '__main__':
    print("正在实例化混合注意力网络...")
    model = HybridAttentionNet(num_classes=3, pretrained=False)
    
    # 模拟输入
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("正在进行前向传播测试...")
    out_logits, out_mask = model(dummy_input)
    
    print(f"✅ 测试成功！")
    print(f"分类输出维度 (Logits): {out_logits.shape}")
    print(f"掩码输出维度 (Mask): {out_mask.shape}")