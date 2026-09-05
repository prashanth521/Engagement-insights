import argparse
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def dir_has_images(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Remove empty class directories from ImageFolder splits to avoid torchvision errors.")
    ap.add_argument("--data_root", required=True, help="Root containing train/val/test splits with class subfolders")
    args = ap.parse_args()

    root = Path(args.data_root)
    removed = 0
    for split in ["train", "val", "test"]:
        sp = root / split
        if not sp.exists():
            continue
        for cls_dir in sp.iterdir():
            if cls_dir.is_dir() and not dir_has_images(cls_dir):
                try:
                    # Remove empty dir
                    cls_dir.rmdir()
                    removed += 1
                    print(f"Removed empty class dir: {cls_dir}")
                except Exception as e:
                    print(f"Could not remove {cls_dir}: {e}")
    print(f"Done. Removed {removed} empty class directories.")


if __name__ == "__main__":
    main()
