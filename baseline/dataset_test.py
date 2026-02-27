from torch.utils.data import DataLoader
from dataset import COVIDFocusDataset # 假设你的 Dataset 类保存在 dataset.py 里

# 只有真正直接运行这个脚本时，才会执行以下代码
if __name__ == '__main__':
    # 1. 实例化 Dataset
    dataset = COVIDFocusDataset(
        base_dir='./dataset', 
        split='Train',
        target_size=(224, 224) 
    )

    # 2. 启动 DataLoader (开启 2 个子进程搬运数据)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # 3. 抽查一个 Batch
    print("正在启动子进程加载数据...")
    images, masks, labels = next(iter(dataloader))
    
    print(f"输入图像形状: {images.shape}")  
    print(f"病灶掩码形状: {masks.shape}")   
    print(f"标签形状: {labels.shape}")
    print("数据加载成功！")