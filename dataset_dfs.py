import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DynamicFocusDataset(Dataset):
    def __init__(self, samples, transform=None, return_mask=False, target_size=(512, 512)):
        """
        Dynamic Focus Search Dataset
        :param samples: List of dicts {'img_path': str, 'lung_mask_path': str, 'inf_mask_path': str, 'label': int}
        :param transform: torchvision transforms
        :param return_mask: Whether to return infection mask (for evaluation)
        :param target_size: Target size for the image (default 512x512)
        """
        self.samples = samples
        self.transform = transform
        self.return_mask = return_mask
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        
        # Read image
        img = cv2.imread(info['img_path'])
        if img is None:
            # Fallback or error
            raise FileNotFoundError(f"Image not found: {info['img_path']}")
        
        # Read and apply lung mask (Physical Focus)
        lung_mask_path = info.get('lung_mask_path')
        if lung_mask_path and os.path.exists(lung_mask_path):
            lung_mask = cv2.imread(lung_mask_path, cv2.IMREAD_GRAYSCALE)
            if lung_mask is not None:
                lung_mask = cv2.resize(lung_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                lung_mask = (lung_mask > 127).astype(np.uint8)
                img = cv2.bitwise_and(img, img, mask=lung_mask)
        
        # Resize to target size (512x512)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Read infection mask if needed
        inf_mask = None
        if self.return_mask:
            inf_mask_path = info.get('inf_mask_path')
            if inf_mask_path and os.path.exists(inf_mask_path):
                inf_mask = cv2.imread(inf_mask_path, cv2.IMREAD_GRAYSCALE)
                if inf_mask is not None:
                    inf_mask = cv2.resize(inf_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                    inf_mask = (inf_mask > 127).astype(np.float32)
            
            if inf_mask is None:
                inf_mask = np.zeros(self.target_size, dtype=np.float32)

        # Convert to RGB and PIL for transforms
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            # Default transform if none provided: ToTensor
            img = transforms.ToTensor()(img)

        label = torch.tensor(info['label'], dtype=torch.long)

        if self.return_mask:
            if not isinstance(inf_mask, torch.Tensor):
                 inf_mask = torch.from_numpy(inf_mask).unsqueeze(0).float()
            return img, inf_mask, label
        else:
            return img, label

def get_all_samples(root_dir, splits=('Train',)):
    """
    Traverse dataset directory to get all samples
    """
    classes = ['COVID-19', 'Non-COVID', 'Normal']
    samples = []
    
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for label, cls_name in enumerate(classes):
            cls_dir = os.path.join(split_dir, cls_name)
            img_dir = os.path.join(cls_dir, 'images')
            lung_mask_dir = os.path.join(cls_dir, 'lung masks')
            inf_mask_dir = os.path.join(cls_dir, 'infection masks')
            
            if not os.path.exists(img_dir):
                continue
                
            for img_name in os.listdir(img_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                samples.append({
                    'img_path': os.path.join(img_dir, img_name),
                    'lung_mask_path': os.path.join(lung_mask_dir, img_name),
                    'inf_mask_path': os.path.join(inf_mask_dir, img_name),
                    'label': label,
                    'id': img_name
                })
    return samples
