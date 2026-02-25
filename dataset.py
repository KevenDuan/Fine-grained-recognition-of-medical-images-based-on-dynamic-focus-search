import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class COVIDFocusDataset(Dataset):
    def __init__(self, base_dir, split='Train', target_size=(224, 224)):
        """
        初始化 COVIDFocus 数据集
        :param base_dir: 数据集根目录
        :param split: 'Train', 'Val', 或 'Test'
        :param target_size: 统一缩放的分辨率，默认 224x224 适配主流网络
        """
        self.base_dir = os.path.join(base_dir, split)
        self.target_size = target_size
        self.classes = ['COVID-19', 'Non-COVID', 'Normal']
        self.data_infos = []
        
        for label, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(self.base_dir, cls_name)
            img_dir = os.path.join(cls_dir, 'images')
            lung_mask_dir = os.path.join(cls_dir, 'lung masks')
            inf_mask_dir = os.path.join(cls_dir, 'infection masks')
            
            if not os.path.exists(img_dir):
                continue
                
            for img_name in os.listdir(img_dir):
                if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                self.data_infos.append({
                    'img_path': os.path.join(img_dir, img_name),
                    'lung_mask_path': os.path.join(lung_mask_dir, img_name),
                    'inf_mask_path': os.path.join(inf_mask_dir, img_name),
                    'label': label
                })
                
    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        info = self.data_infos[idx]
        
        # 1. 读取原图和肺部掩码
        img = cv2.imread(info['img_path'], cv2.IMREAD_GRAYSCALE)
        lung_mask = cv2.imread(info['lung_mask_path'], cv2.IMREAD_GRAYSCALE)
        
        # 2. 物理聚焦：过滤背景干扰
        if lung_mask is not None:
            lung_mask_binary = (lung_mask > 127).astype(np.uint8)
            img = img * lung_mask_binary
            
        # 3. 读取病灶掩码
        if os.path.exists(info['inf_mask_path']):
            inf_mask = cv2.imread(info['inf_mask_path'], cv2.IMREAD_GRAYSCALE)
            inf_mask = (inf_mask > 127).astype(np.float32)
        else:
            inf_mask = np.zeros_like(img, dtype=np.float32)
            
        # 4. 统一尺寸缩放
        # 原图缩放：使用默认的双线性插值，保持图像平滑
        img = cv2.resize(img, self.target_size)
        # 掩码缩放：必须使用最近邻插值(INTER_NEAREST)，防止 0 和 1 的边缘产生模糊的小数！
        inf_mask = cv2.resize(inf_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 5. 格式转换：灰度转RGB，并转换为 PyTorch Tensor 格式 (C, H, W)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        
        # 掩码直接增加一个通道维度 (1, H, W) 并转为 Tensor
        inf_mask_tensor = torch.from_numpy(inf_mask).unsqueeze(0).float()
            
        return img_tensor, inf_mask_tensor, info['label']