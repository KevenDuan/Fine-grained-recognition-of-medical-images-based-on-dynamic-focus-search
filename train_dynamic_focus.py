import os
import torch
import torch.nn as nn
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
from sklearn.model_selection import KFold
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If on Mac with MPS
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    use_amp = device.type == 'cuda'
    
    pbar = tqdm(loader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Mixed precision only for CUDA
        if use_amp:
            with autocast():
                outputs = model(images)
                loss_global = criterion(outputs['pred_global'], labels)
                loss_local = criterion(outputs['pred_local'], labels)
                loss_fused = criterion(outputs['pred_fused'], labels)
                loss = loss_global + loss_local + loss_fused
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
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs['pred_fused'], labels) # Use fused for validation metric
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs['pred_fused'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(outputs['pred_fused'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)

def main():
    set_seed()
    
    # Dataset preparation
    root_dir = 'dataset' # Assuming run from project root
    all_samples = get_all_samples(root_dir)
    print(f"Total samples: {len(all_samples)}")
    
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
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    results = []
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_samples)):
        print(f"FOLD {fold+1}/{NUM_FOLDS}")
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
        
        train_dataset = DynamicFocusDataset(train_samples, transform=train_transform)
        val_dataset = DynamicFocusDataset(val_samples, transform=val_transform)
        test_dataset = DynamicFocusDataset(test_samples, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        # Model
        model = DynamicFocusNet(num_classes=3).to(DEVICE)
        
        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        scaler = GradScaler() # For AMP
        criterion = nn.CrossEntropyLoss()
        
        # EMA
        ema = EMA(model, decay=0.999)
        
        # TensorBoard
        writer = SummaryWriter(f'runs/fold_{fold+1}')
        
        # Training Loop
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ema, DEVICE)
            
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
                torch.save(model.state_dict(), f'runs/fold_{fold+1}/best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break
        
        os.makedirs(f"runs/fold_{fold+1}", exist_ok=True)
        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            save_path=f"runs/fold_{fold+1}/training_curve.png",
        )
        
        # Test on Test Set
        print("Testing on Fold Test Set...")
        # Load best model
        model.load_state_dict(torch.load(f'runs/fold_{fold+1}/best_model.pth'))
        test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)
        print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
        
        results.append({
            'fold': fold + 1,
            'test_acc': test_acc,
            'best_val_acc': best_val_acc
        })
        
        writer.close()
        
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Cross-Validation Completed.")
    print(results)

if __name__ == '__main__':
    main()
