# 动态聚焦搜索（DFS）与消融实验：代码说明与使用手册

本手册面向毕业设计/论文写作场景，系统整理本项目中：
- DFS 训练代码（含指标与曲线输出）
- Baseline 5 折交叉验证代码（与 DFS 同协议可对比）
- 消融实验一键运行与“精美图片”报告生成（柱状图、表格、混淆矩阵、训练曲线拼图、聚焦可视化图集）

适用仓库路径：`Fine-grained-recognition-of-medical-images-based-on-dynamic-focus-search`

---

## 目录

- [1. 目录结构（快速导航）](#1-目录结构快速导航)
- [2. 环境要求与安装](#2-环境要求与安装)
- [3. DFS 算法与实现原理](#3-dfs-算法与实现原理)
  - [3.1 核心代码文件](#31-核心代码文件)
  - [3.2 模型输出与可解释性](#32-模型输出与可解释性)
- [4. 训练与评估（DFS）](#4-训练与评估dfs)
  - [4.1 训练脚本使用方法](#41-训练脚本使用方法)
  - [4.2 参数说明（配置项）](#42-参数说明配置项)
  - [4.3 输出文件说明（返回值）](#43-输出文件说明返回值)
- [5. Baseline 训练与评估（同协议 5 折 CV）](#5-baseline-训练与评估同协议-5-折-cv)
  - [5.1 使用方法](#51-使用方法)
  - [5.2 参数说明（配置项）](#52-参数说明配置项)
  - [5.3 输出文件说明（返回值）](#53-输出文件说明返回值)
- [6. 消融实验：一键运行与报告生成](#6-消融实验一键运行与报告生成)
  - [6.1 消融实验设计（论文写作推荐）](#61-消融实验设计论文写作推荐)
  - [6.2 一键运行脚本](#62-一键运行脚本)
  - [6.3 报告生成脚本（精美图片）](#63-报告生成脚本精美图片)
- [7. 注意事项与潜在问题](#7-注意事项与潜在问题)
- [8. 可运行示例（从训练到报告一条龙）](#8-可运行示例从训练到报告一条龙)

---

## 1. 目录结构（快速导航）

关键文件如下（省略无关文件）：

```text
Fine-grained-recognition-of-medical-images-based-on-dynamic-focus-search/
├── dataset/                          # 数据集目录（Train/Test）
├── baseline/
│   ├── train_baseline.py             # 原 baseline（Train/Val 版）
│   └── train_baseline_cv.py          # 新增：同协议 5 折 CV baseline（可对比）
├── dataset_dfs.py                    # DFS 数据加载（512 输入 + lung mask）
├── model_dfs.py                      # DFS 模型（Global + Local + Attention + TopK 区域）
├── train_dynamic_focus.py            # DFS 训练（StratifiedKFold + Macro-F1/AUC + 曲线）
├── utils_dfs.py                      # EMA、曲线绘图、注意力可视化
├── ablation/
│   ├── run_ablation.py               # 一键跑 baseline + DFS 消融
│   └── make_report.py                # 生成消融报告（多图：柱状/表格/混淆矩阵/图集）
└── requirements.txt                  # 依赖
```

---

## 2. 环境要求与安装

### 环境要求

- Python：建议 3.10/3.11
- PyTorch：支持 CUDA（NVIDIA）或 Apple Silicon（MPS）均可
- 依赖：见 `requirements.txt`

### 安装步骤

推荐使用 conda：

```bash
conda create -n dfs-medical python=3.11 -y
conda activate dfs-medical
pip install -r requirements.txt
```

建议验证：

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available())"
python -c "import cv2, sklearn, matplotlib; print('ok')"
```

---

## 3. DFS 算法与实现原理

DFS（Dynamic Focus Search）的核心思想是：

- **Global 分支（低分辨率全局上下文）**：将输入影像下采样后提取全局特征，建立“粗粒度”判别能力。
- **Attention 定位（病灶关注区域）**：对 Global 特征生成注意力热力图（attention map），定位高响应区域。
- **Local 分支（高分辨率局部细节）**：根据注意力位置从原图裁剪局部 patch，再提取细粒度特征。
- **多尺度融合（Global + Local）**：融合全局与局部特征后输出最终分类预测。

此结构在医学影像中常用于提高细粒度分类、增强可解释性，并提升少数类的召回/宏平均 F1。

### 3.1 核心代码文件

- 数据加载：`dataset_dfs.py`
- 模型结构：`model_dfs.py`
- 训练与评估：`train_dynamic_focus.py`
- EMA 与绘图、可视化：`utils_dfs.py`

### 3.2 模型输出与可解释性

DFS 模型前向输出为一个字典（见 [model_dfs.py](file:///Users/duanhao/Desktop/Fine-grained-recognition-of-medical-images-based-on-dynamic-focus-search/model_dfs.py) 的 `forward` 返回值），包含：

- `pred_fused`：融合分支 logits（最终分类依据）
- `pred_global`：全局分支 logits（辅助监督）
- `pred_local`：局部分支 logits（辅助监督）
- `att_map`：注意力图（B,1,7,7），用于解释性热力图
- `crop_coords`：实际用于 Local 分支裁剪的框（x1,y1,x2,y2）
- `topk_coords` / `topk_weights`：TopK 关键区域坐标及贡献权重（用于“前 3 区域”输出）
- `focus_score`：聚焦置信度（来自 attention map 最大响应）

---

## 4. 训练与评估（DFS）

训练脚本：`train_dynamic_focus.py`

### 4.1 训练脚本使用方法

在项目根目录运行：

```bash
python train_dynamic_focus.py
```

该脚本默认：
- 从 `dataset/Train` 读取训练样本做 **StratifiedKFold**
- 每折内部按 **7:1:2** 切分 train/val/test
- 额外对 `dataset/Test` 做 external test（如果存在）
- 输出每折曲线图、每折 `test_metrics.json`（Acc/Macro-F1/AUC/CM/Report）

### 4.2 参数说明（配置项）

在 `train_dynamic_focus.py` 顶部配置区可以修改：

- `BATCH_SIZE`：批大小（默认 16）
- `NUM_EPOCHS`：最大训练轮数（默认 30）
- `LEARNING_RATE`：学习率（默认 1e-4）
- `WEIGHT_DECAY`：权重衰减（默认 1e-4）
- `NUM_FOLDS`：K 折数量（默认 5）
- `PATIENCE`：早停 patience（默认 10）
- `ACCUMULATION_STEPS`：梯度累积步数（默认 4）
- `NUM_WORKERS`：DataLoader worker 数量
- `ATT_LOSS_WEIGHT`：注意力监督权重（消融关键开关，设为 0 即关闭）
- `FOCUS_MODE`：聚焦策略（消融关键开关）
  - `attn`：注意力峰值聚焦（默认）
  - `center`：固定中心裁剪（消融）
  - `random`：随机裁剪（可选消融）
- `RUNS_ROOT`：输出根目录（默认 `runs`）
- `EXP_NAME`：实验名称（用于消融区分输出目录与汇总 JSON）

#### DynamicFocusNet 参数说明

模型构造函数（`model_dfs.py`）：

```python
DynamicFocusNet(
    num_classes=3,
    pretrained=True,
    input_size=512,
    crop_size=224,
    focus_mode="attn",
    topk=3,
)
```

- `num_classes`：分类类别数（本项目为 3）
- `pretrained`：是否加载 ImageNet 预训练
- `input_size`：输入边长（当前数据加载默认为 512）
- `crop_size`：局部裁剪 patch 边长（默认 224）
- `focus_mode`：聚焦策略（同上）
- `topk`：输出 TopK 关键区域数（默认 3）

### 4.3 输出文件说明（返回值）

DFS 训练会生成如下文件（每折）：

- `{RUNS_ROOT}/{EXP_NAME}/fold_k/best_model.pth`
- `{RUNS_ROOT}/{EXP_NAME}/fold_k/training_curve.png`
- `{RUNS_ROOT}/{EXP_NAME}/fold_k/test_metrics.json`
- `{RUNS_ROOT}/{EXP_NAME}/fold_k/external_test_metrics.json`（若存在 external test）
- 根目录：`training_results_{EXP_NAME}.json`（按折汇总）

`test_metrics.json` 字段示例：

```json
{
  "acc": 0.91,
  "macro_f1": 0.89,
  "auc_ovr_macro": 0.95,
  "confusion_matrix": [[...],[...],[...]],
  "classification_report": { "...": "..." }
}
```

---

## 5. Baseline 训练与评估（同协议 5 折 CV）

Baseline 同协议脚本：`baseline/train_baseline_cv.py`

### 5.1 使用方法

在项目根目录运行：

```bash
python baseline/train_baseline_cv.py
```

它与 DFS **使用一致的协议**：
- `dataset/Train` 上做 StratifiedKFold
- 每折内部 7:1:2 切分
- 输出每折曲线图与 `test_metrics.json`

### 5.2 参数说明（配置项）

在 `baseline/train_baseline_cv.py` 顶部配置区可以修改：

- `BATCH_SIZE / NUM_EPOCHS / LEARNING_RATE / WEIGHT_DECAY`
- `NUM_FOLDS / PATIENCE / ACCUMULATION_STEPS / NUM_WORKERS`
- `IMAGE_SIZE`：baseline 输入尺寸（默认 224）
- `RUNS_ROOT`：输出根目录（默认 `baseline/runs`）
- `EXP_NAME`：实验名称（默认 `baseline`）

### 5.3 输出文件说明（返回值）

Baseline 会生成：

- `{RUNS_ROOT}/{EXP_NAME}/fold_k/best_model.pth`
- `{RUNS_ROOT}/{EXP_NAME}/fold_k/training_curve.png`
- `{RUNS_ROOT}/{EXP_NAME}/fold_k/test_metrics.json`
- 根目录：`baseline_training_results_{EXP_NAME}.json`

---

## 6. 消融实验：一键运行与报告生成

### 6.1 消融实验设计（论文写作推荐）

推荐四组（足以支撑毕业设计“模块有效性”论证）：

1. `baseline`：ResNet-18
2. `dfs_center`：双分支 + 固定中心裁剪（无动态聚焦）
3. `dfs_no_att`：注意力聚焦裁剪，但关闭 attention mask 监督（`ATT_LOSS_WEIGHT=0`）
4. `dfs_full`：完整 DFS（`focus_mode=attn` + `ATT_LOSS_WEIGHT>0`）

### 6.2 一键运行脚本

脚本：`ablation/run_ablation.py`

#### 使用方法

```bash
python ablation/run_ablation.py --root ablation_outputs
```

快速试跑（仅验证流程，不用于论文结论）：

```bash
python ablation/run_ablation.py --root ablation_outputs --fast
```

#### 参数说明

```bash
python ablation/run_ablation.py \
  --root ablation_outputs \
  --fast \
  --experiments baseline dfs_center dfs_no_att dfs_full
```

- `--root`：输出根目录（会创建并写入各实验的 fold 输出）
- `--fast`：快速模式（少 epochs/少 folds，主要用于确认能跑通）
- `--experiments`：选择要跑的实验集合（可多选）

### 6.3 报告生成脚本（精美图片）

脚本：`ablation/make_report.py`

#### 使用方法

```bash
python ablation/make_report.py --root ablation_outputs --out ablation_outputs/report
```

#### 参数说明

```bash
python ablation/make_report.py \
  --root ablation_outputs \
  --out ablation_outputs/report \
  --experiments baseline dfs_center dfs_no_att dfs_full \
  --gallery_per_class 3
```

- `--root`：消融输出根目录（必须包含 `exp_name/fold_k/test_metrics.json`）
- `--out`：报告输出目录
- `--experiments`：参与报告汇总的实验集合
- `--no_gallery`：不生成 DFS 聚焦图集
- `--gallery_per_class`：每类抽取多少张样本生成图集（默认 3）

#### 返回值（输出文件）

报告会生成（可直接放入论文）：

- `metrics_bar.png`：Accuracy/Macro-F1/AUC 的带误差条柱状图
- `summary_table.png`：mean±std 表格图
- `per_class_f1.png`：每类 F1 对比图
- `cm_<exp>.png`：各实验平均混淆矩阵热力图
- `training_curves_montage.png`：训练曲线拼图（使用 fold_1 的曲线作为展示）
- `focus_gallery_<exp>.png`：DFS 实验的聚焦可视化图集（Top3 框 + 权重）

---

## 7. 注意事项与潜在问题

- 训练耗时：5-fold × 30 epochs 在 CPU/MPS 上可能较慢，建议先用 `--fast` 验证流程。
- 聚焦策略的可微性：当前 `attn` 聚焦使用离散 top1 框裁剪，聚焦位置本身不可微；注意力监督（`ATT_LOSS_WEIGHT`）用于增强定位学习。
- 数据集依赖：脚本默认数据目录结构为 `dataset/Train` 与 `dataset/Test`，且每类含 `images/`、`lung masks/`、`infection masks/`（mask 缺失会自动用全零处理）。
- 输出目录：训练会生成大量 `runs/`、`baseline/runs/`、`ablation_outputs/` 文件，已通过 `.gitignore` 忽略，避免误提交。
- 指标可比性：论文中对比必须确保 baseline 与 DFS 使用同一协议（本项目已提供同协议 baseline 脚本）。

---

## 8. 可运行示例（从训练到报告一条龙）

### 示例 1：快速跑通（建议先做）

```bash
python ablation/run_ablation.py --root ablation_outputs --fast
python ablation/make_report.py --root ablation_outputs --out ablation_outputs/report --no_gallery
```

### 示例 2：正式消融（用于论文结论）

```bash
python ablation/run_ablation.py --root ablation_outputs
python ablation/make_report.py --root ablation_outputs --out ablation_outputs/report --gallery_per_class 3
```

完成后，重点查看：

```text
ablation_outputs/report/
├── metrics_bar.png
├── summary_table.png
├── per_class_f1.png
├── cm_baseline.png
├── cm_dfs_full.png
├── training_curves_montage.png
└── focus_gallery_dfs_full.png
```

