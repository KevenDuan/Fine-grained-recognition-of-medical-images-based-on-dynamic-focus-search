import os
# 环境配置：解决本地底层的各种死锁问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import random # 新增：用于随机抽取图片

# 强行设置 Matplotlib 后端，防止 GUI 弹窗导致程序假死
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


def load_trained_model(weight_path, num_classes=3, device='cpu'):
    """加载训练好的 ResNet 模型"""
    print(f"[*] 正在加载模型权重: {weight_path}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件：{weight_path}")
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval() 
    return model


def generate_batch_cam(model, image_dir, mask_dir, save_path, num_images=5, device='cpu'):
    """自动随机抽取多张图片，生成 Grad-CAM 并拼接到一张大图上"""
    print(f"[*] 正在扫描目录: {image_dir}")
    
    # 1. 获取目录下所有的图片文件名
    all_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(all_files) == 0:
        raise ValueError(f"在 {image_dir} 中没有找到任何图片！")
        
    # 如果图片总数不够5张，就有多少拿多少
    actual_num = min(num_images, len(all_files))
    
    # 2. 随机抽取指定数量的图片
    selected_files = random.sample(all_files, actual_num)
    print(f"[*] 成功抽取 {actual_num} 张图片进行分析: \n    {selected_files}")

    # 提取特征层并初始化 Grad-CAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # 3. 动态创建大画板 (行数为抽取图片数，列数为3)
    # 每行高度为 5，宽度为 15
    fig, axes = plt.subplots(actual_num, 3, figsize=(15, 5 * actual_num))
    
    # 如果只有1张图，axes 会变成一维数组，为了代码统一，强行转为二维
    if actual_num == 1:
        axes = [axes]

    # 4. 循环处理每一张抽到的图片
    for i, file_name in enumerate(selected_files):
        img_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)

        # -- 读取与预处理原图 --
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"[-] 读取失败，跳过: {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # -- 生成热力图 --
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        rgb_img_float = img_resized.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

        # -- 读取真实掩码 --
        truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if truth_mask is not None:
            truth_mask = cv2.resize(truth_mask, (224, 224))
        else:
            truth_mask = np.zeros((224, 224), dtype=np.uint8)

        # -- 将三张图画到对应的格子里 --
        axes[i][0].imshow(img_resized)
        axes[i][0].set_title(f'Original: {file_name}')
        axes[i][0].axis('off')
        
        axes[i][1].imshow(cam_image)
        axes[i][1].set_title('AI Search Focus (Grad-CAM)')
        axes[i][1].axis('off')
        
        axes[i][2].imshow(truth_mask, cmap='gray')
        axes[i][2].set_title('Doctor Label (Ground Truth)')
        axes[i][2].axis('off')

    # 5. 调整排版并整体保存
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n🎉 批量分析完成！聚合大图已保存至: {save_path}")


if __name__ == '__main__':
    # ================= 配置区 =================
    DEVICE = torch.device("cpu")
    WEIGHT_PATH = './baselineResult/resnet18_best_model.pth'
    
    # 我们不再指定某一张图片，而是指定整个文件夹的路径
    IMAGE_DIR = 'dataset/Test/COVID-19/images/'
    MASK_DIR = 'dataset/Test/COVID-19/infection masks/'
    
    SAVE_RESULT = 'cam_result_3_images.png'
    NUM_IMAGES_TO_TEST = 3  # 你可以随时改成 10，代码会自动生成更长的对比图！
    # ==========================================

    # 1. 初始化模型
    resnet_model = load_trained_model(WEIGHT_PATH, num_classes=3, device=DEVICE)
    
    # 2. 执行批量自动化生成
    generate_batch_cam(
        model=resnet_model, 
        image_dir=IMAGE_DIR, 
        mask_dir=MASK_DIR, 
        save_path=SAVE_RESULT, 
        num_images=NUM_IMAGES_TO_TEST,
        device=DEVICE
    )