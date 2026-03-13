import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("TensorBoard not found. Using DummyWriter.")
    class SummaryWriter:
        def __init__(self, log_dir=None):
            pass
        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
            pass
        def close(self):
            pass
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import time
import json

from dataset_dfs import DynamicFocusDataset, get_all_samples
from model_dfs import DynamicFocusNet
from utils_dfs import EMA, plot_training_curves

# Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 30 # Baseline was 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_FOLDS = 5
PATIENCE = 10
ACCUMULATION_STEPS = 4 # Gradient accumulation
NUM_WORKERS = 4
ATT_LOSS_WEIGHT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If on Mac with MPS
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_one_epoch(model, loader, criterion, att_criterion, optimizer, scaler, ema, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    use_amp = device.type == 'cuda'
    
    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        if len(batch) == 3:
            images, inf_masks, labels = batch
            inf_masks = inf_masks.to(device, non_blocking=use_amp)
        else:
            images, labels = batch
            inf_masks = None
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)
        
        # Mixed precision only for CUDA
        if use_amp:
            with autocast():
                outputs = model(images)
                loss_global = criterion(outputs['pred_global'], labels)
                loss_local = criterion(outputs['pred_local'], labels)
                loss_fused = criterion(outputs['pred_fused'], labels)
                loss = loss_global + loss_local + loss_fused
                if inf_masks is not None:
                    att_map = outputs['att_map']
                    mask_small = F.interpolate(inf_masks, size=att_map.shape[-2:], mode='nearest')
                    mask_small = (mask_small > 0.5).to(att_map.dtype)
                    loss_att = att_criterion(att_map, mask_small)
                    loss = loss + ATT_LOSS_WEIGHT * loss_att
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()
        else:
            outputs = model(images)
            loss_global = criterion(outputs['pred_global'], labels)
            loss_local = criterion(outputs['pred_local'], labels)
            loss_fused = criterion(outputs['pred_fused'], labels)
            loss = loss_global + loss_local + loss_fused
            if inf_masks is not None:
                att_map = outputs['att_map']
                mask_small = F.interpolate(inf_masks, size=att_map.shape[-2:], mode='nearest')
                mask_small = (mask_small > 0.5).to(att_map.dtype)
                loss_att = att_criterion(att_map, mask_small)
                loss = loss + ATT_LOSS_WEIGHT * loss_att
            loss = loss / ACCUMULATION_STEPS
            
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update()
        
        running_loss += loss.item() * ACCUMULATION_STEPS

        
        # Calculate accuracy on fused prediction
        _, predicted = torch.max(outputs['pred_fused'], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (i + 1), 'acc': correct / total})
        
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Metrics for F1, AUC
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            non_blocking = device.type == 'cuda'
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            
            outputs = model(images)
            loss = criterion(outputs['pred_fused'], labels) # Use fused for validation metric
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs['pred_fused'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(outputs['pred_fused'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)

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

def main():
    set_seed()

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: cuda ({gpu_name})")
    else:
        print(f"Device: {DEVICE.type}")
    
    # Dataset preparation
    project_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(project_dir, 'dataset')
    all_samples = get_all_samples(root_dir, splits=('Train',))
    external_test_samples = get_all_samples(root_dir, splits=('Test',))
    print(f"Total samples: {len(all_samples)}")
    if len(all_samples) == 0:
        raise RuntimeError(f"No samples found under: {root_dir}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # K-Fold
    num_folds = min(NUM_FOLDS, len(all_samples))
    if num_folds < 2:
        raise RuntimeError(f"Need at least 2 samples for K-Fold (found {len(all_samples)})")
    strat_labels = np.array([s['label'] for s in all_samples])
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    results = []
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(np.zeros(len(all_samples)), strat_labels)):
        print(f"FOLD {fold+1}/{num_folds}")
        print("--------------------------------")
        
        # Split train into train/val (7:1 ratio from 80% total -> 70% train, 10% val)
        # train_ids contains 80% of data. We need to split it 7:1.
        # 7/8 of train_ids for training, 1/8 for validation.
        num_train = len(train_ids)
        split_idx = int(num_train * 7 / 8)
        
        # Shuffle train_ids before splitting (KFold already shuffled, but let's be safe)
        np.random.shuffle(train_ids)
        
        real_train_ids = train_ids[:split_idx]
        val_ids = train_ids[split_idx:]
        
        # Create Subsets
        train_samples = [all_samples[i] for i in real_train_ids]
        val_samples = [all_samples[i] for i in val_ids]
        test_samples = [all_samples[i] for i in test_ids]
        
        train_dataset = DynamicFocusDataset(train_samples, transform=train_transform, return_mask=True)
        val_dataset = DynamicFocusDataset(val_samples, transform=val_transform)
        test_dataset = DynamicFocusDataset(test_samples, transform=val_transform)
        external_test_dataset = DynamicFocusDataset(external_test_samples, transform=val_transform) if len(external_test_samples) > 0 else None
        
        pin_memory = DEVICE.type == 'cuda'
        persistent_workers = NUM_WORKERS > 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
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
                prefetch_factor=2,
            )
        
        # Model
        model = DynamicFocusNet(num_classes=3).to(DEVICE)
        
        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        scaler = GradScaler() # For AMP
        criterion = nn.CrossEntropyLoss()
        att_criterion = nn.BCELoss()
        
        # EMA
        ema = EMA(model, decay=0.999)
        
        # TensorBoard
        run_dir = os.path.join(project_dir, 'runs', f'fold_{fold+1}')
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(run_dir)
        
        # Training Loop
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, att_criterion, optimizer, scaler, ema, DEVICE)
            
            # Validation (using EMA weights)
            ema.apply_shadow()
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
            ema.restore()
            
            scheduler.step()
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Time: {time.time()-start_time:.1f}s")
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Logging
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Checkpointing
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
        
        # Test on Test Set
        print("Testing on Fold Test Set...")
        # Load best model
        model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
        test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)
        print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
        test_metrics = compute_metrics(test_preds, test_labels, num_classes=3)
        with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Fold {fold+1} Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        external_test_acc = None
        external_test_metrics = None
        if external_test_loader is not None:
            print("Testing on External Test Set...")
            _, external_test_acc, external_test_preds, external_test_labels = validate(model, external_test_loader, criterion, DEVICE)
            external_test_metrics = compute_metrics(external_test_preds, external_test_labels, num_classes=3)
            with open(os.path.join(run_dir, 'external_test_metrics.json'), 'w') as f:
                json.dump(external_test_metrics, f, indent=2)
            print(f"External Test Accuracy: {external_test_acc:.4f}")
            print(f"External Test Macro-F1: {external_test_metrics['macro_f1']:.4f}")
        
        results.append({
            'fold': fold + 1,
            'test_acc': test_acc,
            'external_test_acc': external_test_acc,
            'best_val_acc': best_val_acc,
            'test_macro_f1': test_metrics['macro_f1'],
            'external_test_macro_f1': external_test_metrics['macro_f1'] if external_test_metrics is not None else None,
        })
        
        writer.close()
        
    # Save results
    with open(os.path.join(project_dir, 'training_results.json'), 'w') as f:
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
    print(results)

if __name__ == '__main__':
    main()
