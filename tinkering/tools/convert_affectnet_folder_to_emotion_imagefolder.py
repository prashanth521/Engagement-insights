import argparse
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

EXPECTED_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
]

# Some Kaggle drops also include 'contempt'. Map it to 'disgust' by default.
EMOTION_ALIASES: Dict[str, str] = {
    "contempt": "disgust",
}


def find_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Convert a folder-structured AffectNet (Kaggle) into an ImageFolder for emotion pretraining.")
    parser.add_argument("--images_root", type=str, required=True, help="Root that contains class subfolders (recursively)")
    parser.add_argument("--out_root", type=str, default="tinkering/data/processed_emotion", help="Output root")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    images_root = Path(args.images_root)
    out_root = Path(args.out_root)

    # Build class set by walking and reading parent directory names
    imgs = find_images(images_root)
    if not imgs:
        print("No images found under images_root.")
        return

    # Determine class from nearest parent directory name that matches expected or alias
    def normalize_label_from_path(p: Path) -> str:
        for parent in [p.parent, *p.parents]:
            name = parent.name.strip().lower()
            if name in EXPECTED_EMOTIONS:
                return name
            if name in EMOTION_ALIASES:
                return EMOTION_ALIASES[name]
        return ""

    # Collect items (path, class)
    items: List[tuple[Path, str]] = []
    for p in imgs:
        cls = normalize_label_from_path(p)
        if cls:
            items.append((p, cls))

    if not items:
        print("Could not infer any classes from folder names. Ensure images live under emotion-named folders.")
        return

    classes = EXPECTED_EMOTIONS
    for sp in ["train", "val", "test"]:
        for c in classes:
            (out_root / sp / c).mkdir(parents=True, exist_ok=True)

    n = len(items)
    t_end = int(n * args.train_ratio)
    v_end = t_end + int(n * args.val_ratio)

    def which_split(i: int) -> str:
        if i < t_end:
            return "train"
        if i < v_end:
            return "val"
        return "test"

    saved, skipped = 0, 0
    for i, (in_path, cls) in enumerate(tqdm(items, desc="Converting (folder->emotion ImageFolder)")):
        sp = which_split(i)
        try:
            with Image.open(in_path).convert("RGB") as img:
                img = img.resize((args.img_size, args.img_size))
                out_path = out_root / sp / cls / f"{i:08d}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path, quality=95)
                saved += 1
        except Exception:
            skipped += 1
    print(f"Done. Saved: {saved}, Skipped: {skipped}. Output: {out_root}")


if __name__ == "__main__":
    main()
