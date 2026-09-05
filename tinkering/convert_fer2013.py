import argparse
import csv
import os
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

# FER-2013 emotions: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
# Map to engagement labels (heuristic, for demo; real mapping deserves care or supervised relabeling)
EMO_TO_ENGAGE = {
    3: "attentive",   # Happy -> engaged/attentive
    6: "attentive",   # Neutral -> attentive (weak assumption)
    5: "confused",    # Surprise -> confused
    2: "confused",    # Fear -> confused
    4: "distracted",  # Sad -> distracted/bored
    0: "distracted",  # Angry -> distracted
    1: "distracted",  # Disgust -> distracted
}


def save_image_from_pixels(pixels: str, out_path: Path, size=(48, 48)):
    arr = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
    arr = arr.reshape(size)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def convert(csv_path: str, out_dir: str, split_filter: str = ""):
    out = Path(out_dir)
    total = 0
    used = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            total += 1
            if split_filter and row.get("Usage", "").lower() != split_filter.lower():
                continue
            emo = int(row["emotion"])
            label = EMO_TO_ENGAGE.get(emo)
            if label is None:
                continue
            used += 1
            pixels = row["pixels"]
            out_path = out / label / f"fer_{i:07d}.jpg"
            save_image_from_pixels(pixels, out_path)
    print(f"Wrote {used} images (from {total} rows) to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path to fer2013.csv")
    parser.add_argument("out", type=str, help="Output directory for ImageFolder structure")
    parser.add_argument("--usage", type=str, default="", help="Optional filter by Usage column: Training/PublicTest/PrivateTest")
    args = parser.parse_args()
    convert(args.csv, args.out, args.usage)


if __name__ == "__main__":
    main()
