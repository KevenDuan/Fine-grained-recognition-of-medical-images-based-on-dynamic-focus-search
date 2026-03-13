# 消融实验说明

## 实验目的

本目录包含了基于动态聚焦搜索的医疗影像细粒度识别项目的消融实验模型。通过对比不同注意力机制的效果，验证动态聚焦搜索的有效性。

## 实验模型

### 1. BaselineResNet
- **描述**：基础ResNet-18模型，无注意力机制
- **用途**：作为消融实验的基线模型
- **文件**：`baseline_resnet.py`

### 2. ChannelAttentionNet
- **描述**：仅使用通道注意力机制的模型
- **用途**：验证通道注意力对模型性能的影响
- **文件**：`channel_attention_net.py`

### 3. HybridAttentionNet
- **描述**：同时使用通道注意力和空间注意力机制的模型
- **用途**：验证混合注意力机制的效果
- **文件**：`hybrid_attention_net.py`

### 4. DynamicFocusNet（主模型）
- **描述**：使用空间注意力机制的动态聚焦网络（位于项目根目录）
- **用途**：作为最终的动态聚焦搜索模型
- **文件**：`../dynamic_focus_net.py`

## 训练脚本

- **文件**：`train_ablation.py`
- **功能**：统一训练所有消融实验模型，并生成训练曲线和结果

## 运行方法

1. **确保数据集已准备好**：
   - 数据集应位于 `../dataset` 目录
   - 包含 Train、Val、Test 三个子目录

2. **安装依赖**：
   ```bash
   pip install -r ../requirements.txt
   ```

3. **运行训练脚本**：
   ```bash
   cd ablation
   python train_ablation.py
   ```

4. **查看结果**：
   - 训练完成后，模型权重将保存在 `ablation_results` 目录
   - 训练曲线将保存在 `ablation_results` 目录
   - 终端会显示各模型的最佳准确率

## 实验结果分析

通过对比不同模型的性能，可以分析：

1. **注意力机制的有效性**：对比 BaselineResNet 和其他注意力模型
2. **不同注意力机制的效果**：对比 ChannelAttentionNet 和 HybridAttentionNet
3. **动态聚焦搜索的优势**：对比所有模型与 DynamicFocusNet

## 预期结果

- 带有注意力机制的模型应优于基线模型
- 同时使用通道和空间注意力的模型应优于仅使用通道注意力的模型
- DynamicFocusNet 应具有最佳性能，证明动态聚焦搜索的有效性