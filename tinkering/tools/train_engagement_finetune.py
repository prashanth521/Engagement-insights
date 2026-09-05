import argparse
import json
from pathlib import Path
from typing import List
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure project root (parent of tools/) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model

TARGET_CLASSES = ["attentive", "distracted", "confused", "disengaged"]


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.RandomAffine(degrees=8, translate=(0.02, 0.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def load_pretrained_backbone(model: torch.nn.Module, ckpt_path: Path) -> None:
    data = torch.load(str(ckpt_path), map_location="cpu")
    state = data.get("model_state", data)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained weights with missing={len(missing)} unexpected={len(unexpected)}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        b = x.size(0)
        loss_sum += loss.item() * b
        acc_sum += accuracy(logits.detach(), y) * b
        n += b
    return loss_sum / max(1, n), acc_sum / max(1, n)


def evaluate(model, loader, criterion, device, num_classes: int, class_names: List[str]):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    # Confusion matrix
    import numpy as np
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            b = x.size(0)
            loss_sum += loss.item() * b
            acc_sum += accuracy(logits, y) * b
            preds = logits.argmax(dim=1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                cm[int(t.item()), int(p.item())] += 1
            n += b
    # Per-class precision/recall
    prec = []
    rec = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec.append(precision)
        rec.append(recall)
    print("Validation confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("Per-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name:12} | precision={prec[i]:.3f} recall={rec[i]:.3f}")
    return loss_sum / max(1, n), acc_sum / max(1, n)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune engagement classifier on 4-class ImageFolder dataset.")
    parser.add_argument("--data_root", type=str, required=True, help="Root with train/val/test and 4 class folders")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrained_ckpt", type=str, default="", help="Optional checkpoint from emotion pretraining")
    parser.add_argument("--out", type=str, default="tinkering/checkpoints/engagement_finetune.pt")
    parser.add_argument("--save_latest", action="store_true", help="Also save a 'latest' checkpoint every epoch")
    parser.add_argument("--class-weights", type=str, default="none", choices=["none", "auto"], help="Use class-weighted loss (auto computes from train set)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, eval_tf = get_transforms(args.img_size)

    # Expect folder names to match TARGET_CLASSES; will map by alphabetical order of folders in ImageFolder
    train_ds = datasets.ImageFolder(str(Path(args.data_root) / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(Path(args.data_root) / "val"), transform=eval_tf)

    # Optional check: ensure classes align with our target set
    expected = TARGET_CLASSES
    actual = train_ds.classes
    print(f"Train classes: {actual}")
    if actual != expected:
        print("WARNING: Class order differs from expected. The saved checkpoint will record 'classes' as detected here.")

    workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=(workers>0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=(workers>0))

    model = create_model(num_classes=len(train_ds.classes), pretrained=True, arch=args.arch).to(device)
    if args.pretrained_ckpt:
        load_pretrained_backbone(model, Path(args.pretrained_ckpt))

    # Class weights
    if args.class_weights == "auto":
        # Count samples per class in the training set
        import numpy as np
        counts = np.zeros(len(train_ds.classes), dtype=np.int64)
        for _, y in train_ds.samples:
            counts[y] += 1
        # Inverse frequency weights
        weights = (1.0 / np.maximum(counts, 1)).astype(np.float32)
        # Normalize to mean 1.0 for stability
        weights = weights * (len(weights) / weights.sum())
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print("Using class weights:", weights.tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, num_classes=len(train_ds.classes), class_names=train_ds.classes)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")
        # Save latest every epoch if requested
        if args.save_latest:
            latest_path = Path(args.out).with_suffix("")
            latest_path = Path(str(latest_path) + "_latest.pt")
            ckpt_latest = {
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": args.img_size,
                "arch": args.arch,
            }
            torch.save(ckpt_latest, latest_path)
            print(f"Saved latest checkpoint to {latest_path}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = {
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": args.img_size,
                "arch": args.arch,
            }
            torch.save(ckpt, args.out)
            print(f"Saved best checkpoint to {args.out}")

    print("Done.")


if __name__ == "__main__":
    main()
