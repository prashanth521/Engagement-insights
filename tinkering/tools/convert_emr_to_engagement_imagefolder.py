import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

# Target classes and canonical order for your project
TARGET_CLASSES = ["attentive", "distracted", "confused", "disengaged"]

mp_face_mesh = mp.solutions.face_mesh


@dataclass
class CropConfig:
    pad: int = 20
    min_side: int = 128


def read_emr_csv(csv_path: Path, img_col: str, label_col: Optional[str], score_col: Optional[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if img_col not in row:
                continue
            if label_col is None and score_col is None:
                continue
            rows.append(row)
    return rows


def normalize_label(lbl: str) -> Optional[str]:
    s = (lbl or "").strip().lower()
    if s in {"engaged", "engaging", "high", "attentive"}:
        return "attentive"
    if s in {"bored", "low", "offtask", "distracted"}:
        return "distracted"
    if s in {"confused", "confusion"}:
        return "confused"
    if s in {"sleepy", "sleep", "drowsy", "disengaged", "very_low", "asleep"}:
        return "disengaged"
    return None


def map_score_to_label(score: float, thr_att: float, thr_dist: float, thr_dis: float) -> str:
    # score in [0,1], higher = more engaged
    if score >= thr_att:
        return "attentive"
    if score >= thr_dist:
        return "confused"  # mid scores likely attentive-but-struggling
    if score >= thr_dis:
        return "distracted"
    return "disengaged"


def ensure_dirs(root: Path, splits: List[str]) -> None:
    for sp in splits:
        for c in TARGET_CLASSES:
            (root / sp / c).mkdir(parents=True, exist_ok=True)


def compute_bbox_from_landmarks(landmarks: np.ndarray, w: int, h: int, pad: int) -> Tuple[int, int, int, int]:
    xs = (landmarks[:, 0] * w).astype(np.int32)
    ys = (landmarks[:, 1] * h).astype(np.int32)
    x1, y1 = xs.min() - pad, ys.min() - pad
    x2, y2 = xs.max() + pad, ys.max() + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    return x1, y1, x2, y2


def is_blurry(bgr: np.ndarray, thr: float) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < thr


def process_image(in_path: Path, out_path: Path, img_size: int, use_mediapipe: bool, crop_cfg: CropConfig, blur_thr: float) -> bool:
    try:
        bgr = cv2.imread(str(in_path))
        if bgr is None:
            return False
        if is_blurry(bgr, blur_thr):
            return False
        if use_mediapipe:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as fm:
                res = fm.process(rgb)
                if not res.multi_face_landmarks:
                    return False
                lm = res.multi_face_landmarks[0]
                pts = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
                h, w = bgr.shape[:2]
                x1, y1, x2, y2 = compute_bbox_from_landmarks(pts, w, h, crop_cfg.pad)
                crop = bgr[y1:y2, x1:x2]
        else:
            crop = bgr
        if crop.size == 0:
            return False
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb).resize((img_size, img_size))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=95)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EMR to 4-class engagement ImageFolder.")
    parser.add_argument("--emr_csv", type=str, required=True, help="CSV with at least image path and label or score")
    parser.add_argument("--images_root", type=str, required=True, help="Root where image paths are relative (or absolute in CSV)")
    parser.add_argument("--out_root", type=str, default="tinkering/data/processed_engagement", help="Output root")
    parser.add_argument("--img_col", type=str, default="path")
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--score_col", type=str, default=None, help="If provided, map scores to labels")
    parser.add_argument("--thr_att", type=float, default=0.65)
    parser.add_argument("--thr_dist", type=float, default=0.45)
    parser.add_argument("--thr_dis", type=float, default=0.20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--use_mediapipe", action="store_true", help="Enable face crop via MediaPipe FaceMesh")
    parser.add_argument("--blur_thr", type=float, default=120.0)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--min_side", type=int, default=128)
    args = parser.parse_args()

    if args.label_col is None and args.score_col is None:
        print("You must provide either --label_col or --score_col")
        return

    rows = read_emr_csv(Path(args.emr_csv), args.img_col, args.label_col, args.score_col)
    if not rows:
        print("No items read. Check CSV and column names.")
        return

    out_root = Path(args.out_root)
    ensure_dirs(out_root, ["train", "val", "test"])

    n = len(rows)
    idxs = list(range(n))
    idxs.sort()
    t_end = int(n * args.train_ratio)
    v_end = t_end + int(n * args.val_ratio)
    def which_split(i: int) -> str:
        if i < t_end:
            return "train"
        if i < v_end:
            return "val"
        return "test"

    images_root = Path(args.images_root)
    crop_cfg = CropConfig(pad=args.pad, min_side=args.min_side)

    ok, skipped = 0, 0
    for i, row in enumerate(tqdm(rows, desc="Converting EMR")):
        rel = row.get(args.img_col, "").strip()
        if not rel:
            skipped += 1
            continue
        in_path = Path(rel)
        if not in_path.is_absolute():
            in_path = images_root / in_path

        label: Optional[str] = None
        if args.label_col is not None:
            label = normalize_label(row.get(args.label_col, ""))
        if label is None and args.score_col is not None:
            try:
                s = float(row.get(args.score_col, ""))
                label = map_score_to_label(s, args.thr_att, args.thr_dist, args.thr_dis)
            except Exception:
                label = None
        if label not in TARGET_CLASSES:
            skipped += 1
            continue

        sp = which_split(i)
        out_path = out_root / sp / label / f"{i:08d}.jpg"
        if process_image(in_path, out_path, args.img_size, args.use_mediapipe, crop_cfg, args.blur_thr):
            ok += 1
        else:
            skipped += 1

    print(f"Done. Saved: {ok}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
