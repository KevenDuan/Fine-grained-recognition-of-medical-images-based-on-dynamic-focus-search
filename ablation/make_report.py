import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


CLASSES = ['COVID-19', 'Non-COVID', 'Normal']


def _read_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def _safe_mean_std(values):
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None, None
    return float(arr.mean()), float(arr.std())


def _load_fold_metrics(root_dir: Path, exp_name: str):
    fold_dirs = sorted([p for p in root_dir.joinpath(exp_name).glob('fold_*') if p.is_dir()])
    per_fold = []
    for fd in fold_dirs:
        p = fd / 'test_metrics.json'
        if not p.exists():
            continue
        per_fold.append(_read_json(p))
    return per_fold


def _extract_per_class_f1(report_dict):
    out = []
    for i in range(len(CLASSES)):
        key = str(i)
        if key in report_dict:
            out.append(float(report_dict[key].get('f1-score', 0.0)))
        else:
            out.append(0.0)
    return out


def _save_bar_with_error(ax, labels, means, stds, title, ylim=None):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color='#4C78A8', alpha=0.9, edgecolor='#2F4B7C', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_confusion_matrix(cm, title, save_path: Path):
    cm = np.array(cm, dtype=np.float64)
    fig = plt.figure(figsize=(6, 5), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=25, ha='right')
    ax.set_yticklabels(CLASSES)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.0f}', ha='center', va='center', color='black', fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _montage(images, cols, bg=(255, 255, 255), pad=12):
    if len(images) == 0:
        return None
    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    cell_w = max(widths)
    cell_h = max(heights)
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new('RGB', (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), bg)
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad) + (cell_w - im.size[0]) // 2
        y = pad + r * (cell_h + pad) + (cell_h - im.size[1]) // 2
        canvas.paste(im, (x, y))
    return canvas


def _collect_training_curves(root_dir: Path, exp_names):
    images = []
    for exp in exp_names:
        p = root_dir / exp / 'fold_1' / 'training_curve.png'
        if p.exists():
            images.append(Image.open(p).convert('RGB'))
    return images


def _sample_images(dataset_root: Path, per_class=3):
    out = []
    for cls in CLASSES:
        p = dataset_root / 'Test' / cls / 'images'
        if not p.exists():
            continue
        imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        for x in imgs[:per_class]:
            out.append((cls, x))
    return out


def _overlay_focus(image_tensor, att_map, boxes, weights, mean, std, alpha=0.4):
    import cv2
    import torch

    img = image_tensor.detach().float().cpu()
    mean_t = torch.tensor(mean, dtype=img.dtype).view(1, 1, 3)
    std_t = torch.tensor(std, dtype=img.dtype).view(1, 1, 3)
    img = img.permute(1, 2, 0)
    img = img * std_t + mean_t
    img = img.clamp(0, 1)
    img = (img.numpy() * 255).astype(np.uint8).copy()

    att = att_map.squeeze().detach().float().cpu().numpy()
    att = cv2.resize(att, (img.shape[1], img.shape[0]))
    att = att - att.min()
    att = att / (att.max() + 1e-6)
    att_u8 = (att * 255).astype(np.uint8)
    heat = cv2.applyColorMap(att_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 1.0 - alpha, heat, alpha, 0)

    colors = [(0, 255, 0), (0, 215, 255), (255, 0, 0)]
    for i, b in enumerate(boxes[:3]):
        x1, y1, x2, y2 = map(int, b)
        c = colors[i % len(colors)]
        cv2.rectangle(blended, (x1, y1), (x2, y2), c, 2)
        w = weights[i] if i < len(weights) else 0.0
        cv2.putText(
            blended,
            f'{i+1}:{w:.3f}',
            (x1 + 4, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            c,
            2,
            cv2.LINE_AA,
        )
    return blended


def _make_focus_gallery(root_dir: Path, exp_name: str, out_dir: Path, dataset_root: Path, fold='fold_1', per_class=3):
    import cv2
    import torch
    import torchvision.transforms as transforms

    ckpt = root_dir / exp_name / fold / 'best_model.pth'
    if not ckpt.exists():
        return

    from model_dfs import DynamicFocusNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')

    focus_mode = 'center' if exp_name == 'dfs_center' else 'attn'
    model = DynamicFocusNet(num_classes=3, pretrained=False, focus_mode=focus_mode).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    samples = _sample_images(dataset_root, per_class=per_class)
    ims = []
    for cls, img_path in samples:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = tfm(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        att = out['att_map'][0]
        boxes = out.get('topk_coords', [[out['crop_coords'][0]]])[0]
        weights = out.get('topk_weights', [[]])[0]
        overlay = _overlay_focus(x[0], att, boxes, weights, mean=mean, std=std, alpha=0.42)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.putText(
            overlay,
            f'{exp_name} | {cls}',
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ims.append(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)))

    montage = _montage(ims, cols=3)
    if montage is not None:
        montage.save(out_dir / f'focus_gallery_{exp_name}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='ablation_outputs')
    parser.add_argument('--out', default='ablation_outputs/report')
    parser.add_argument(
        '--experiments',
        nargs='*',
        default=['baseline', 'dfs_center', 'dfs_no_att', 'dfs_full'],
    )
    parser.add_argument('--no_gallery', action='store_true')
    parser.add_argument('--gallery_per_class', type=int, default=3)
    args = parser.parse_args()

    root_dir = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_class = {}
    cms = {}
    for exp in args.experiments:
        per_fold = _load_fold_metrics(root_dir, exp)
        if len(per_fold) == 0:
            continue

        accs = []
        f1s = []
        aucs = []
        cm_sum = None
        per_class_f1s = []
        for fm in per_fold:
            accs.append(float(fm.get('acc', 0.0)))
            f1s.append(float(fm.get('macro_f1', 0.0)))
            aucs.append(fm.get('auc_ovr_macro', None))
            cm = fm.get('confusion_matrix', None)
            if cm is not None:
                cm_arr = np.array(cm, dtype=np.float64)
                cm_sum = cm_arr if cm_sum is None else (cm_sum + cm_arr)
            rep = fm.get('classification_report', {})
            per_class_f1s.append(_extract_per_class_f1(rep))

        acc_mean, acc_std = _safe_mean_std(accs)
        f1_mean, f1_std = _safe_mean_std(f1s)
        auc_mean, auc_std = _safe_mean_std(aucs)
        rows.append({
            'exp': exp,
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'macro_f1_mean': f1_mean,
            'macro_f1_std': f1_std,
            'auc_mean': auc_mean,
            'auc_std': auc_std,
            'folds': len(per_fold),
        })
        if cm_sum is not None:
            cms[exp] = (cm_sum / float(len(per_fold))).tolist()
        per_class[exp] = np.mean(np.array(per_class_f1s, dtype=np.float64), axis=0).tolist()

    rows = sorted(rows, key=lambda r: args.experiments.index(r['exp']) if r['exp'] in args.experiments else 999)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(rows, f, indent=2)

    if len(rows) == 0:
        return

    labels = [r['exp'] for r in rows]
    acc_means = [r['acc_mean'] for r in rows]
    acc_stds = [r['acc_std'] for r in rows]
    f1_means = [r['macro_f1_mean'] for r in rows]
    f1_stds = [r['macro_f1_std'] for r in rows]
    auc_means = [r['auc_mean'] if r['auc_mean'] is not None else 0.0 for r in rows]
    auc_stds = [r['auc_std'] if r['auc_std'] is not None else 0.0 for r in rows]

    fig = plt.figure(figsize=(13, 4), dpi=200)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    _save_bar_with_error(ax1, labels, acc_means, acc_stds, 'Accuracy', ylim=(0, 1))
    _save_bar_with_error(ax2, labels, f1_means, f1_stds, 'Macro-F1', ylim=(0, 1))
    _save_bar_with_error(ax3, labels, auc_means, auc_stds, 'AUC-ROC (OVR, Macro)', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out_dir / 'metrics_bar.png')
    plt.close(fig)

    table_data = []
    for r in rows:
        table_data.append([
            r['exp'],
            f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}",
            f"{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}",
            "NA" if r['auc_mean'] is None else f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}",
        ])
    fig = plt.figure(figsize=(10.5, 0.55 + 0.35 * len(table_data)), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    tbl = ax.table(
        cellText=table_data,
        colLabels=['Experiment', 'Accuracy', 'Macro-F1', 'AUC'],
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'summary_table.png')
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    width = 0.8 / max(1, len(labels))
    x = np.arange(len(CLASSES))
    for i, exp in enumerate(labels):
        vals = per_class.get(exp, [0.0] * len(CLASSES))
        ax.bar(x + i * width, vals, width=width, label=exp)
    ax.set_xticks(x + width * (len(labels) - 1) / 2.0)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1)
    ax.set_title('Per-class F1 (mean across folds)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'per_class_f1.png')
    plt.close(fig)

    for exp, cm in cms.items():
        _plot_confusion_matrix(cm, f'Confusion Matrix (avg) - {exp}', out_dir / f'cm_{exp}.png')

    curves = _collect_training_curves(root_dir, labels)
    montage = _montage(curves, cols=min(2, max(1, len(curves))))
    if montage is not None:
        montage.save(out_dir / 'training_curves_montage.png')

    if not args.no_gallery:
        dataset_root = Path(__file__).resolve().parents[1] / 'dataset'
        for exp in labels:
            if exp.startswith('dfs'):
                _make_focus_gallery(
                    root_dir=root_dir,
                    exp_name=exp,
                    out_dir=out_dir,
                    dataset_root=dataset_root,
                    per_class=max(1, int(args.gallery_per_class)),
                )


if __name__ == '__main__':
    main()
