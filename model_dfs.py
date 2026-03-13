import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return self.sigmoid(out)

class DynamicFocusNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(DynamicFocusNet, self).__init__()
        # Load ResNet18 backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove average pooling and fc layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Attention module
        self.attention = AttentionBlock(512)
        
        # Classifier
        # Global feature: 512, Local feature: 512 -> Concat: 1024
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Auxiliary classifiers for deep supervision
        self.cls_global = nn.Linear(512, num_classes)
        self.cls_local = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (B, 3, 512, 512)
        batch_size = x.size(0)
        
        # 1. Global Branch
        # Downsample to 224x224 for global view
        x_global = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        
        # Extract global features
        f_global = self.features(x_global) # (B, 512, 7, 7)
        
        # Generate Attention Map
        att_map = self.attention(f_global) # (B, 1, 7, 7)
        
        # 2. Focus Mechanism (Crop)
        # Find peak activation in attention map
        # Flatten: (B, 49)
        att_flat = att_map.view(batch_size, -1)
        val, idx = torch.max(att_flat, dim=1)
        
        # Convert index to (h, w) coordinates in 7x7 grid
        h_idx = idx // 7
        w_idx = idx % 7
        
        # Map to original 512x512 space
        # Grid size 7x7, Image size 224x224 -> Stride 32
        # But we mapped 512->224. So relative to 512, stride is 32 * (512/224) approx 73?
        # Let's think in normalized coordinates.
        # Center of grid cell (h, w): (h + 0.5)/7, (w + 0.5)/7
        
        # We want to crop a 224x224 patch from 512x512 image.
        # Half size = 112.
        crop_size = 224
        half_crop = crop_size // 2
        
        # Center coordinates in 512 image
        center_y = (h_idx.float() + 0.5) / 7.0 * 512
        center_x = (w_idx.float() + 0.5) / 7.0 * 512
        
        # Clamp centers to ensure crop is within bounds
        # min center = 112, max center = 512 - 112 = 400
        center_y = torch.clamp(center_y, min=half_crop, max=512-half_crop)
        center_x = torch.clamp(center_x, min=half_crop, max=512-half_crop)
        
        # Perform crop for each image in batch
        x_local_list = []
        top_left_coords = [] # Store for visualization
        
        for i in range(batch_size):
            cy = int(center_y[i].item())
            cx = int(center_x[i].item())
            
            y1 = cy - half_crop
            y2 = cy + half_crop
            x1 = cx - half_crop
            x2 = cx + half_crop
            
            crop = x[i:i+1, :, y1:y2, x1:x2]
            x_local_list.append(crop)
            top_left_coords.append((x1, y1, x2, y2))
            
        x_local = torch.cat(x_local_list, dim=0) # (B, 3, 224, 224)
        
        # 3. Local Branch
        f_local = self.features(x_local) # (B, 512, 7, 7)
        
        # 4. Fusion
        # Global Pooling
        feat_g = F.adaptive_avg_pool2d(f_global, (1, 1)).view(batch_size, -1)
        feat_l = F.adaptive_avg_pool2d(f_local, (1, 1)).view(batch_size, -1)
        
        # Concatenate
        feat_fused = torch.cat([feat_g, feat_l], dim=1)
        
        # Predictions
        pred_global = self.cls_global(feat_g)
        pred_local = self.cls_local(feat_l)
        pred_fused = self.classifier(feat_fused)
        
        return {
            'pred_fused': pred_fused,
            'pred_global': pred_global,
            'pred_local': pred_local,
            'att_map': att_map,
            'crop_coords': top_left_coords,
            'feat_fused': feat_fused
        }
