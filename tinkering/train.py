import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

from model import create_model


LABELS_4 = ["attentive", "distracted", "confused", "disengaged"]


def build_transforms(img_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def load_imagefolder(path: str, transform):
    ds = ImageFolder(root=path, transform=transform)
    return ds


def prepare_dataloaders(data_dirs: list, batch_size: int, val_split: float, num_workers: int, img_size: int):
    train_tf, val_tf = build_transforms(img_size)

    datasets = [load_imagefolder(p, train_tf) for p in data_dirs]
    # Ensure class sets match
    class_sets = [tuple(ds.classes) for ds in datasets]
    if len(set(class_sets)) != 1:
        raise ValueError(f"All datasets must share identical classes and ordering. Got: {class_sets}")
    classes = datasets[0].classes

    full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    total_len = len(full)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full, [train_len, val_len])

    # random_split returns Subset; switch transform for validation by modifying underlying dataset(s)
    # For simplicity, wrap via DataLoader-level transform using collate? Instead, rely on train_tf being okay for val
    # To keep distinct, rebuild val dataset from the same dirs with val_tf and index selection
    if isinstance(full, ConcatDataset):
        # Build a flat list of (dataset_index, sample_index) for val subset
        # Then create a lightweight Subset over a combined val dataset with val transform
        val_datasets = [load_imagefolder(p, val_tf) for p in data_dirs]
        val_full = ConcatDataset(val_datasets)
        val_indices = val_ds.indices  # type: ignore[attr-defined]
        from torch.utils.data import Subset
        val_ds_final = Subset(val_full, val_indices)
    else:
        val_ds_final = ImageFolder(root=data_dirs[0], transform=val_tf)
        # Keep only val indices
        from torch.utils.data import Subset
        val_ds_final = Subset(val_ds_final, val_ds.indices)  # type: ignore[attr-defined]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds_final, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, classes


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return epoch_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss_sum += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def save_checkpoint(state: Dict, ckpt_path: Path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs='+', help="One or more roots with class subfolders (ImageFolder). You can pass FER+ and AffectNet dirs together.")
    parser.add_argument("--out", type=str, default="checkpoints/engagement_resnet.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet50"], help="Backbone CNN")
    parser.add_argument("--dml", action="store_true", help="Use DirectML backend if available (for AMD/Intel GPUs on Windows)")
    args = parser.parse_args()

    # Device selection: prefer CUDA, then optional DirectML, else CPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.dml:
        try:
            import torch_directml
            device = torch_directml.device()
            print("Using DirectML device for training")
        except Exception as e:
            print(f"DirectML requested but not available: {e}. Falling back to CPU.")

    # Resolve output path: if relative, place under script directory
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    train_loader, val_loader, classes = prepare_dataloaders(args.data_dirs, args.batch, args.val_split, args.workers, args.img_size)
    num_classes = len(classes)

    model = create_model(num_classes=num_classes, pretrained=True, arch=args.arch)
    model.to(device)

    # Handle class imbalance with weighted loss
    from collections import Counter
    import numpy as np
    
    # Count class samples
    class_counts = Counter()
    for _, target in train_loader.dataset:
        class_counts[target] += 1
    
    # Calculate class weights (inverse frequency)
    total_samples = sum(class_counts.values())
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts.values()])
    class_weights = class_weights.to(device)
    
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # AMP only for CUDA at present; keep disabled for CPU/DML
    scaler = torch.cuda.amp.GradScaler() if (not args.no_amp and torch.cuda.is_available()) else None

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "arch": args.arch,
            }, out_path)
            print(f"Saved best checkpoint to {out_path} (acc={best_val_acc:.3f})")


if __name__ == "__main__":
    main()
