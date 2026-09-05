import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

# Optional: If AffectNet images are not tightly cropped, you can enable face detection/cropping later.
# For now, we keep it simple and just resize to the target size.

# Canonical list of AffectNet emotions we will support for pretraining
EXPECTED_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
]


def read_labels_csv(csv_path: Path, img_col: str, label_col: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get(img_col, "").strip()
            if not img_path:
                continue
            raw = str(row.get(label_col, "")).strip()
            if raw == "":
                continue
            # Accept either integer ids or string names
            lbl: str
            try:
                # If it's an int id in [0..6], map to expected name order
                idx = int(raw)
                mapping = {0: "neutral", 1: "happy", 2: "sad", 3: "surprise", 4: "fear", 5: "disgust", 6: "anger"}
                lbl = mapping.get(idx, "")
            except Exception:
                lbl = raw.lower()
            if not lbl:
                continue
            items.append((img_path, lbl))
    return items


def ensure_dirs(root: Path, classes: List[str], splits: List[str]) -> None:
    for sp in splits:
        for c in classes:
            (root / sp / c).mkdir(parents=True, exist_ok=True)


def split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(n))
    # simple deterministic split for reproducibility
    idxs.sort()
    t_end = int(n * train_ratio)
    v_end = t_end + int(n * val_ratio)
    return idxs[:t_end], idxs[t_end:v_end], idxs[v_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AffectNet labels to ImageFolder for emotion pretraining.")
    parser.add_argument("--labels_csv", type=str, required=True, help="CSV with columns for image path and label (name or id)")
    parser.add_argument("--images_root", type=str, required=True, help="Root where image paths are relative to (or absolute in CSV)")
    parser.add_argument("--out_root", type=str, default="tinkering/data/processed_emotion", help="Output root directory")
    parser.add_argument("--img_col", type=str, default="pth", help="CSV column name for image path")
    parser.add_argument("--label_col", type=str, default="label", help="CSV column for label (string or integer id)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    labels = read_labels_csv(Path(args.labels_csv), args.img_col, args.label_col)
    if not labels:
        print("No items read from CSV. Check column names and file.")
        return

    classes = EXPECTED_EMOTIONS
    out_root = Path(args.out_root)
    ensure_dirs(out_root, classes, ["train", "val", "test"])

    n = len(labels)
    train_idx, val_idx, test_idx = split_indices(n, args.train_ratio, args.val_ratio)
    index_to_split: Dict[int, str] = {}
    for i in train_idx:
        index_to_split[i] = "train"
    for i in val_idx:
        index_to_split[i] = "val"
    for i in test_idx:
        index_to_split[i] = "test"

    images_root = Path(args.images_root)
    ok, skipped = 0, 0
    for i, (rel_path, lbl_name) in enumerate(tqdm(labels, desc="Converting")):
        cls = lbl_name.lower()
        if cls not in classes:
            skipped += 1
            continue
        sp = index_to_split.get(i, "train")
        in_path = Path(rel_path)
        if not in_path.is_absolute():
            in_path = images_root / in_path
        try:
            with Image.open(in_path).convert("RGB") as img:
                img = img.resize((args.img_size, args.img_size))
                out_path = out_root / sp / cls / f"{i:08d}.jpg"
                img.save(out_path, quality=95)
                ok += 1
        except Exception:
            skipped += 1
            continue

    print(f"Done. Saved: {ok}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
