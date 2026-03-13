import torch
import torch.nn as nn
import torchvision.models as models

class BaselineResNet(nn.Module):
    """
    基础ResNet模型（无注意力机制）
    消融实验的基线模型
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(BaselineResNet, self).__init__()
        
        # 使用预训练的ResNet-18作为特征提取器
        resnet = models.resnet18(pretrained=pretrained)
        
        # 替换最后的全连接层以适应3分类任务
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)
        
        self.model = resnet
    
    def forward(self, x):
        """
        前向传播
        输入 x: (Batch, 3, 224, 224) 的X光片
        返回: (Batch, num_classes) 的分类结果
        """
        # 直接通过ResNet模型
        logits = self.model(x)
        
        # 为了与其他模型保持一致的输出格式，返回logits和一个空掩码
        # 空掩码的形状与其他模型的注意力掩码相同
        dummy_mask = torch.zeros(x.size(0), 1, 224, 224, device=x.device)
        
        return logits, dummy_mask

# 测试模型
if __name__ == '__main__':
    print("正在实例化基础ResNet模型...")
    model = BaselineResNet(num_classes=3, pretrained=False)
    
    # 模拟输入
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("正在进行前向传播测试...")
    out_logits, out_mask = model(dummy_input)
    
    print(f"✅ 测试成功！")
    print(f"分类输出维度 (Logits): {out_logits.shape}")
    print(f"掩码输出维度 (Mask): {out_mask.shape}")