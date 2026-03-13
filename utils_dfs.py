import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def compute_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.savefig(save_path)
    plt.close()

def visualize_attention(img_tensor, att_map, crop_coords, save_path):
    """
    Visualize original image, attention map, and crop box.
    img_tensor: (3, H, W) normalized
    att_map: (1, 7, 7)
    crop_coords: (x1, y1, x2, y2)
    """
    # Denormalize image
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    
    # Resize attention map to image size
    att = att_map.squeeze().cpu().numpy()
    att = cv2.resize(att, (img.shape[1], img.shape[0]))
    att = (att * 255).astype(np.uint8)
    att_heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
    
    # Superimpose
    heatmap_img = cv2.addWeighted(img, 0.6, att_heatmap, 0.4, 0)
    
    # Draw crop box
    x1, y1, x2, y2 = crop_coords
    cv2.rectangle(heatmap_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Concatenate
    final_img = np.hstack((img, heatmap_img))
    cv2.imwrite(save_path, final_img)
