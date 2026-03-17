import os
import sys
import time
import json
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, log_dir=None):
            pass
        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
            pass
        def close(self):
            pass

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from utils_dfs import EMA, plot_training_curves

BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_FOLDS = 5
PATIENCE = 10
ACCUMULATION_STEPS = 4
NUM_WORKERS = 4
IMAGE_SIZE = 224
RUNS_ROOT = os.path.join('baseline', 'runs')
EXP_NAME = 'baseline'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')

CLASSES = ['COVID-19', 'Non-COVID', 'Normal']


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_all_samples(root_dir, splits=('Train',)):
    samples = []
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
        for label, cls_name in enumerate(CLASSES):
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
                    'id': img_name,
                })
    return samples


class BaselineDataset(Dataset):
    def __init__(self, samples, transform=None, target_size=(224, 224)):
        self.samples = samples
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]

        img = cv2.imread(info['img_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(info['img_path'])

        lung_mask_path = info.get('lung_mask_path')
        if lung_mask_path and os.path.exists(lung_mask_path):
            lung_mask = cv2.imread(lung_mask_path, cv2.IMREAD_GRAYSCALE)
            if lung_mask is not None:
                lung_mask_binary = (lung_mask > 127).astype(np.uint8)
                img = img * lung_mask_binary

        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = torch.tensor(info['label'], dtype=torch.long)
        return img, label


def _softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def compute_metrics(logits, labels, num_classes=3):
    pred = np.argmax(logits, axis=1)
    acc = float((pred == labels).mean())
    macro_f1 = float(f1_score(labels, pred, average='macro'))
    cm = confusion_matrix(labels, pred, labels=list(range(num_classes))).tolist()
    report = classification_report(labels, pred, output_dict=True, zero_division=0)
    auc = None
    try:
        prob = _softmax_np(logits, axis=1)
        auc = float(roc_auc_score(labels, prob, multi_class='ovr', average='macro'))
    except Exception:
        auc = None
    return {
        'acc': acc,
        'macro_f1': macro_f1,
        'auc_ovr_macro': auc,
        'confusion_matrix': cm,
        'classification_report': report,
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'

    pbar = tqdm(loader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        non_blocking = device.type == 'cuda'
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update()

        running_loss += loss.item() * ACCUMULATION_STEPS
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': running_loss / (i + 1), 'acc': correct / total})

    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            non_blocking = device.type == 'cuda'
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_logits.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    return running_loss / len(loader), correct / total, np.array(all_logits), np.array(all_labels)


def main():
    set_seed()

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: cuda ({gpu_name})")
    else:
        print(f"Device: {DEVICE.type}")

    dataset_root = os.path.join(PROJECT_DIR, 'dataset')
    all_samples = get_all_samples(dataset_root, splits=('Train',))
    external_test_samples = get_all_samples(dataset_root, splits=('Test',))
    print(f"Total train samples: {len(all_samples)}")
    if len(all_samples) == 0:
        raise RuntimeError(f"No samples found under: {dataset_root}")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_folds = min(NUM_FOLDS, len(all_samples))
    if num_folds < 2:
        raise RuntimeError(f"Need at least 2 samples for K-Fold (found {len(all_samples)})")

    strat_labels = np.array([s['label'] for s in all_samples])
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    results = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(np.zeros(len(all_samples)), strat_labels)):
        print(f"FOLD {fold+1}/{num_folds}")
        print("--------------------------------")

        num_train = len(train_ids)
        split_idx = int(num_train * 7 / 8)
        np.random.shuffle(train_ids)
        real_train_ids = train_ids[:split_idx]
        val_ids = train_ids[split_idx:]

        train_samples = [all_samples[i] for i in real_train_ids]
        val_samples = [all_samples[i] for i in val_ids]
        test_samples = [all_samples[i] for i in test_ids]

        train_dataset = BaselineDataset(train_samples, transform=train_transform, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        val_dataset = BaselineDataset(val_samples, transform=val_transform, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        test_dataset = BaselineDataset(test_samples, transform=val_transform, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        external_test_dataset = BaselineDataset(external_test_samples, transform=val_transform, target_size=(IMAGE_SIZE, IMAGE_SIZE)) if len(external_test_samples) > 0 else None

        pin_memory = DEVICE.type == 'cuda'
        persistent_workers = NUM_WORKERS > 0
        dataloader_extra = {}
        if NUM_WORKERS > 0:
            dataloader_extra['prefetch_factor'] = 2

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **dataloader_extra,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **dataloader_extra,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **dataloader_extra,
        )
        external_test_loader = None
        if external_test_dataset is not None:
            external_test_loader = DataLoader(
                external_test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                **dataloader_extra,
            )

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss()
        ema = EMA(model, decay=0.9)

        run_dir = os.path.join(PROJECT_DIR, RUNS_ROOT, EXP_NAME, f'fold_{fold+1}')
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(run_dir)

        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ema, DEVICE)

            ema.apply_shadow()
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
            ema.restore()

            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"Time: {time.time()-start_time:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break

        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            save_path=os.path.join(run_dir, 'training_curve.png'),
        )

        model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth'), map_location=DEVICE))

        print("Testing on Fold Test Set...")
        _, test_acc, test_logits, test_labels = validate(model, test_loader, criterion, DEVICE)
        test_metrics = compute_metrics(test_logits, test_labels, num_classes=3)
        with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
        print(f"Fold {fold+1} Test Macro-F1: {test_metrics['macro_f1']:.4f}")

        external_test_acc = None
        external_test_metrics = None
        if external_test_loader is not None:
            print("Testing on External Test Set...")
            _, external_test_acc, external_test_logits, external_test_labels = validate(model, external_test_loader, criterion, DEVICE)
            external_test_metrics = compute_metrics(external_test_logits, external_test_labels, num_classes=3)
            with open(os.path.join(run_dir, 'external_test_metrics.json'), 'w') as f:
                json.dump(external_test_metrics, f, indent=2)
            print(f"External Test Accuracy: {external_test_acc:.4f}")
            print(f"External Test Macro-F1: {external_test_metrics['macro_f1']:.4f}")

        results.append({
            'fold': fold + 1,
            'test_acc': float(test_acc),
            'external_test_acc': float(external_test_acc) if external_test_acc is not None else None,
            'best_val_acc': float(best_val_acc),
            'test_macro_f1': float(test_metrics['macro_f1']),
            'external_test_macro_f1': float(external_test_metrics['macro_f1']) if external_test_metrics is not None else None,
        })

        writer.close()

    summary_path = os.path.join(PROJECT_DIR, f'baseline_training_results_{EXP_NAME}.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("Cross-Validation Completed.")
    test_accs = [r['test_acc'] for r in results]
    print(f"Fold Test Acc mean±std: {np.mean(test_accs):.4f}±{np.std(test_accs):.4f}")
    test_f1s = [r['test_macro_f1'] for r in results]
    print(f"Fold Test Macro-F1 mean±std: {np.mean(test_f1s):.4f}±{np.std(test_f1s):.4f}")
    ext_accs = [r['external_test_acc'] for r in results if r['external_test_acc'] is not None]
    if len(ext_accs) > 0:
        print(f"External Test Acc mean±std: {np.mean(ext_accs):.4f}±{np.std(ext_accs):.4f}")
    ext_f1s = [r['external_test_macro_f1'] for r in results if r['external_test_macro_f1'] is not None]
    if len(ext_f1s) > 0:
        print(f"External Test Macro-F1 mean±std: {np.mean(ext_f1s):.4f}±{np.std(ext_f1s):.4f}")


if __name__ == '__main__':
    main()
