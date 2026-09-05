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


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            b = x.size(0)
            loss_sum += loss.item() * b
            acc_sum += accuracy(logits, y) * b
            n += b
    return loss_sum / max(1, n), acc_sum / max(1, n)


def main():
    parser = argparse.ArgumentParser(description="Pretrain on emotion ImageFolder (e.g., AffectNet-derived)")
    parser.add_argument("--data_root", type=str, required=True, help="Root containing train/val/test subfolders of emotion classes")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="tinkering/checkpoints/pretrain_emotion.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, eval_tf = get_transforms(args.img_size)

    train_ds = datasets.ImageFolder(str(Path(args.data_root) / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(Path(args.data_root) / "val"), transform=eval_tf)

    classes: List[str] = train_ds.classes

    workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=(workers>0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=(workers>0))

    model = create_model(num_classes=len(classes), pretrained=True, arch=args.arch).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = {
                "model_state": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "arch": args.arch,
            }
            torch.save(ckpt, args.out)
            print(f"Saved best checkpoint to {args.out}")

    print("Done.")


if __name__ == "__main__":
    main()
