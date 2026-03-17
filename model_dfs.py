import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1)
        # 🚨 删掉 self.sigmoid = nn.Sigmoid() 这一行

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out  # 🚨 删掉 self.sigmoid()，直接返回 out

class DynamicFocusNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, input_size=512, crop_size=224, focus_mode='attn', topk=3):
        super(DynamicFocusNet, self).__init__()
        self.input_size = int(input_size)
        self.crop_size = int(crop_size)
        self.focus_mode = str(focus_mode)
        self.topk = int(topk)
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
        batch_size = x.size(0)
        input_h = x.size(2)
        input_w = x.size(3)
        
        # 1. Global Branch
        # Downsample to 224x224 for global view
        x_global = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        
        # Extract global features
        f_global = self.features(x_global) # (B, 512, 7, 7)
        
        # Generate Attention Map
        att_map = self.attention(f_global) # (B, 1, 7, 7)
        
        # 2. Focus Mechanism (Crop)
        att_flat = att_map.view(batch_size, -1)
        topk = max(1, self.topk)
        topk_vals, topk_idx = torch.topk(att_flat, k=min(topk, att_flat.size(1)), dim=1)
        idx = topk_idx[:, 0]
        val = topk_vals[:, 0]

        h_idx = idx // att_map.size(-1)
        w_idx = idx % att_map.size(-1)

        crop_size = self.crop_size
        half_crop = crop_size // 2

        if self.focus_mode == 'center':
            center_y = torch.full((batch_size,), float(input_h) / 2.0, device=x.device, dtype=torch.float32)
            center_x = torch.full((batch_size,), float(input_w) / 2.0, device=x.device, dtype=torch.float32)
        elif self.focus_mode == 'random':
            cy = torch.rand((batch_size,), device=x.device, dtype=torch.float32) * float(input_h)
            cx = torch.rand((batch_size,), device=x.device, dtype=torch.float32) * float(input_w)
            center_y = cy
            center_x = cx
        else:
            grid_h = att_map.size(-2)
            grid_w = att_map.size(-1)
            center_y = (h_idx.float() + 0.5) / float(grid_h) * float(input_h)
            center_x = (w_idx.float() + 0.5) / float(grid_w) * float(input_w)

        center_y = torch.clamp(center_y, min=float(half_crop), max=float(input_h - half_crop))
        center_x = torch.clamp(center_x, min=float(half_crop), max=float(input_w - half_crop))
        
        # Perform crop for each image in batch
        x_local_list = []
        top_left_coords = [] # Store for visualization
        topk_boxes = []
        topk_weights = []
        
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

            boxes_i = []
            weights_i = []
            for k in range(topk_idx.size(1)):
                kk = topk_idx[i, k]
                hh = int((kk // att_map.size(-1)).item())
                ww = int((kk % att_map.size(-1)).item())
                cyk = (float(hh) + 0.5) / float(att_map.size(-2)) * float(input_h)
                cxk = (float(ww) + 0.5) / float(att_map.size(-1)) * float(input_w)
                cyk = float(np.clip(cyk, half_crop, input_h - half_crop))
                cxk = float(np.clip(cxk, half_crop, input_w - half_crop))
                y1k = int(cyk - half_crop)
                y2k = int(cyk + half_crop)
                x1k = int(cxk - half_crop)
                x2k = int(cxk + half_crop)
                boxes_i.append((x1k, y1k, x2k, y2k))
                weights_i.append(float(topk_vals[i, k].item()))
            topk_boxes.append(boxes_i)
            topk_weights.append(weights_i)
            
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
            'topk_coords': topk_boxes,
            'topk_weights': topk_weights,
            'focus_score': val.detach().cpu().tolist(),
            'feat_fused': feat_fused
        }
