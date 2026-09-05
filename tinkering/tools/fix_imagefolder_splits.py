import argparse
import shutil
from pathlib import Path
from typing import List

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def list_images(d: Path) -> List[Path]:
    files: List[Path] = []
    if not d.exists():
        return files
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files


def ensure_min_per_class(root: Path, min_val: int, min_test: int, copy_only: bool = True) -> None:
    train_root = root / "train"
    val_root = root / "val"
    test_root = root / "test"

    classes = [d.name for d in train_root.iterdir() if d.is_dir()]

    for cls in classes:
        tr_dir = train_root / cls
        va_dir = val_root / cls
        te_dir = test_root / cls
        va_dir.mkdir(parents=True, exist_ok=True)
        te_dir.mkdir(parents=True, exist_ok=True)

        tr_imgs = list_images(tr_dir)
        va_imgs = list_images(va_dir)
        te_imgs = list_images(te_dir)

        # Top up val
        need_val = max(0, min_val - len(va_imgs))
        if need_val > 0:
            take = tr_imgs[:need_val]
            for i, src in enumerate(take):
                dst = va_dir / f"fix_val_{i:04d}{src.suffix.lower()}"
                if copy_only:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
            print(f"Class {cls}: added {len(take)} to val")
            tr_imgs = tr_imgs[need_val:]

        # Top up test
        need_test = max(0, min_test - len(te_imgs))
        if need_test > 0:
            take = tr_imgs[:need_test]
            for i, src in enumerate(take):
                dst = te_dir / f"fix_test_{i:04d}{src.suffix.lower()}"
                if copy_only:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
            print(f"Class {cls}: added {len(take)} to test")


def main():
    ap = argparse.ArgumentParser(description="Ensure each class in an ImageFolder has at least N images in val and test by copying from train.")
    ap.add_argument("--data_root", required=True, help="Root containing train/val/test")
    ap.add_argument("--min_val", type=int, default=10)
    ap.add_argument("--min_test", type=int, default=10)
    ap.add_argument("--move", action="store_true", help="Move instead of copy from train")
    args = ap.parse_args()

    root = Path(args.data_root)
    ensure_min_per_class(root, args.min_val, args.min_test, copy_only=(not args.move))
    print("Done fixing splits.")


if __name__ == "__main__":
    main()
