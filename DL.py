# -*- coding: utf-8 -*-
"""
CIFAR-10 training script:
- Augment + normalize
- Train/val split (stratified)
- Compact CNN baseline (Conv-BN-ReLU, MaxPool, Dropout, Global AvgPool)
- Accuracy/loss logging + plots
- Test evaluation
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# ------------------------------
# Reproducibility & device
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.deterministic = False  # better speed
    cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ------------------------------
# Transforms
# ------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ------------------------------
# Datasets & stratified split
# ------------------------------
os.makedirs("data", exist_ok=True)

# We create two train datasets with different transforms (train vs val)
full_train_ds_for_split = CIFAR10(root="data", train=True, download=True, transform=None)
y_all = np.array(full_train_ds_for_split.targets)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros_like(y_all), y_all))

train_ds = CIFAR10(root="data", train=True, download=False, transform=train_tfms)
val_ds   = CIFAR10(root="data", train=True, download=False, transform=test_tfms)  # no aug

train_ds = Subset(train_ds, train_idx)
valid_ds = Subset(val_ds,   val_idx)

test_ds  = CIFAR10(root="data", train=False, download=True, transform=test_tfms)

# ------------------------------
# DataLoaders
# ------------------------------
num_workers = 2 if device == "cuda" else 0
pin_memory = True if device == "cuda" else False

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# ------------------------------
# Model
# ------------------------------
class CIFARSmallNet(nn.Module):
    def __init__(self, num_classes=10, p_drop=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # 3x32x32
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # -> 32x16x16
            nn.Dropout(p_drop),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # -> 64x8x8
            nn.Dropout(p_drop),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)  # -> 128x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),            # 128
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CIFARSmallNet(num_classes=10).to(device)

# Optional: model summary
try:
    from torchinfo import summary
    print(summary(model, input_size=(32, 3, 32, 32),
                  col_names=['input_size', 'output_size', 'num_params', 'trainable']))
except Exception as e:
    print("[WARN] torchinfo not available or failed; skipping summary.", e)

# ------------------------------
# Loss, Optimizer, Scheduler
# ------------------------------
loss_fn  = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# ------------------------------
# Train / Validate
# ------------------------------
from tqdm.auto import tqdm

def train(model, loss_fn, optimizer, scheduler, train_loader, valid_loader, epochs=30):
    results = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += y.size(0)

        # ---- Validate ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.inference_mode():
            for X, y in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = loss_fn(logits, y)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)

        # Aggregates
        train_loss /= len(train_loader)
        val_loss   /= len(valid_loader)
        train_acc   = train_correct / train_total
        val_acc     = val_correct / val_total

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)

        print(f"[INFO] Epoch {epoch+1:03d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4%} | Val Acc: {val_acc:.4%}")

    return results

results = train(model, loss_fn, optimizer, scheduler, train_loader, valid_loader, epochs=100)

# ------------------------------
# Test
# ------------------------------
def test(model, loss_fn, test_loader):
    model.eval()
    with torch.inference_mode():
        test_loss, test_correct, test_total = 0.0, 0, 0
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss   += loss.item()
            test_correct += (logits.argmax(dim=1) == y).sum().item()
            test_total   += y.size(0)
    test_loss /= len(test_loader)
    test_acc   = test_correct / test_total
    print(f"[INFO] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4%}")
    return {'test_loss': test_loss, 'test_acc': test_acc}

test_results = test(model, loss_fn, test_loader)

# ------------------------------
# Plot curves
# ------------------------------
def plot_results(results):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(results['train_loss'], label='Train Loss')
    ax[0].plot(results['val_loss'],   label='Val Loss')
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].grid(True); ax[0].legend()

    ax[1].plot(results['train_acc'], label='Train Acc')
    ax[1].plot(results['val_acc'],   label='Val Acc')
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].grid(True); ax[1].legend()
    plt.tight_layout(); plt.show()

plot_results(results)
