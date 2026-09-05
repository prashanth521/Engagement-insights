import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Optional import of mediapipe for face cropping
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    mp = None
    mp_face_mesh = None

# Canonical 4-class schema used by this project
TARGET_CLASSES = ["attentive", "distracted", "confused", "disengaged"]


@dataclass
class CropConfig:
    pad: int = 20
    min_side: int = 128


def ensure_dirs(root: Path, splits: List[str]) -> None:
    for sp in splits:
        for c in TARGET_CLASSES:
            (root / sp / c).mkdir(parents=True, exist_ok=True)


def is_blurry(bgr: np.ndarray, thr: float) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < thr


def compute_bbox_from_landmarks(landmarks: np.ndarray, w: int, h: int, pad: int) -> Tuple[int, int, int, int]:
    xs = (landmarks[:, 0] * w).astype(np.int32)
    ys = (landmarks[:, 1] * h).astype(np.int32)
    x1, y1 = xs.min() - pad, ys.min() - pad
    x2, y2 = xs.max() + pad, ys.max() + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    return x1, y1, x2, y2


def label_from_daisee_row(row: Dict[str, str],
                           conf_thr: int,
                           frus_thr: int,
                           bor_thr: int,
                           eng_att: int,
                           eng_dist: int) -> Optional[str]:
    """
    Map DAiSEE per-frame/per-clip labels to the 4-class schema.
    Expected numeric levels 0 (very low) .. 3 (very high) for these columns if present:
      - engagement, boredom, confusion, frustration

    Priority:
      1) confusion >= conf_thr => "confused"
      2) engagement >= eng_att => "attentive"
      3) engagement <= 0 => "disengaged"
      4) boredom >= bor_thr => "disengaged"
      5) frustration >= frus_thr => "distracted"
      6) engagement < eng_dist => "distracted"
      else => "attentive"
    """
    def to_int(val: Optional[str]) -> Optional[int]:
        try:
            return int(val) if val is not None and val != "" else None
        except Exception:
            try:
                # Some CSVs may store floats; cast to int
                return int(float(val)) if val is not None and val != "" else None
            except Exception:
                return None

    engagement = to_int(row.get("engagement"))
    boredom = to_int(row.get("boredom"))
    confusion = to_int(row.get("confusion"))
    frustration = to_int(row.get("frustration"))

    # 1) High confusion dominates
    if confusion is not None and confusion >= conf_thr:
        return "confused"

    # 2) High engagement => attentive
    if engagement is not None and engagement >= eng_att:
        return "attentive"

    # 3) Very low engagement => disengaged
    if engagement is not None and engagement <= 0:
        return "disengaged"

    # 4) High boredom => disengaged (or distracted). We choose disengaged.
    if boredom is not None and boredom >= bor_thr:
        return "disengaged"

    # 5) High frustration => distracted (off-task/struggling)
    if frustration is not None and frustration >= frus_thr:
        return "distracted"

    # 6) Low engagement => distracted
    if engagement is not None and engagement < eng_dist:
        return "distracted"

    # Fallback
    return "attentive"


def iter_labels_csv(labels_csv: Path) -> Iterable[Dict[str, str]]:
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def guess_split(row: Dict[str, str], default_split: str) -> str:
    for key in ["split", "usage", "set", "partition"]:
        if key in row and row[key]:
            s = row[key].strip().lower()
            if s in {"train", "training"}:
                return "train"
            if s in {"val", "valid", "validation"}:
                return "val"
            if s in {"test", "testing"}:
                return "test"
    return default_split


def resolve_video_path(row: Dict[str, str], root: Path, videos_root: Optional[Path]) -> Optional[Path]:
    # Common columns that may reference the video
    for key in ["video", "video_path", "path", "filename", "file"]:
        if key in row and row[key]:
            p = Path(row[key])
            if p.is_absolute():
                return p if p.exists() else None
            # relative: can be relative to videos_root or root
            if videos_root is not None and (videos_root / p).exists():
                return (videos_root / p)
            if (root / p).exists():
                return (root / p)
    # Fallback: if videos_root provided and row has id columns
    vid_id = row.get("video_id") or row.get("vid")
    if videos_root is not None and vid_id:
        # attempt simple patterns
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            candidate = videos_root / f"{vid_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def extract_and_save_frame(bgr: np.ndarray,
                           out_path: Path,
                           img_size: int,
                           use_mediapipe: bool,
                           crop_cfg: CropConfig,
                           blur_thr: float) -> bool:
    try:
        if bgr is None:
            return False
        if is_blurry(bgr, blur_thr):
            return False
        crop = bgr
        if use_mediapipe and mp_face_mesh is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as fm:
                res = fm.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0]
                    pts = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
                    h, w = bgr.shape[:2]
                    x1, y1, x2, y2 = compute_bbox_from_landmarks(pts, w, h, crop_cfg.pad)
                    crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb).resize((img_size, img_size))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=95)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert DAiSEE videos/labels to 4-class ImageFolder.")
    parser.add_argument("--root", type=str, required=True, help="DAiSEE root directory (or a generic root containing labels/videos)")
    parser.add_argument("--labels_csv", type=str, default="", help="Path to labels CSV (if not under root). Must contain engagement/boredom/confusion/frustration columns and a video reference")
    parser.add_argument("--videos_root", type=str, default="", help="Explicit path to videos root if not under --root")
    parser.add_argument("--out_root", type=str, default="tinkering/data/daisee_imagefolder", help="Output root for ImageFolder")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=30, help="Extract every Nth frame if CSV doesn't specify a frame index")
    parser.add_argument("--frame_col", type=str, default="frame", help="Column name for frame index if available")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Default split if not found in labels")
    parser.add_argument("--use_mediapipe", action="store_true", help="Enable face crop via MediaPipe FaceMesh")
    parser.add_argument("--blur_thr", type=float, default=120.0)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--min_side", type=int, default=128)
    # Mapping thresholds
    parser.add_argument("--conf_thr", type=int, default=2)
    parser.add_argument("--frus_thr", type=int, default=2)
    parser.add_argument("--bor_thr", type=int, default=2)
    parser.add_argument("--eng_att", type=int, default=2, help="engagement >= this -> attentive")
    parser.add_argument("--eng_dist", type=int, default=1, help="engagement < this -> distracted")
    args = parser.parse_args()

    root = Path(args.root)
    labels_csv_path: Optional[Path] = Path(args.labels_csv) if args.labels_csv else None
    videos_root = Path(args.videos_root) if args.videos_root else None

    # Try to auto-locate labels CSV if not provided
    if labels_csv_path is None:
        candidates = list(root.rglob("*labels*.csv")) + list(root.rglob("*Labels*.csv")) + list(root.rglob("labels.csv"))
        if candidates:
            labels_csv_path = candidates[0]
            print(f"Auto-selected labels CSV: {labels_csv_path}")
        else:
            print("Error: Could not find labels CSV automatically. Please pass --labels_csv.")
            return

    if videos_root is None:
        # Common directories: Videos/, video/, clips/
        for name in ["Videos", "videos", "clips", "data", "DAiSEE"]:
            candidate = root / name
            if candidate.exists():
                videos_root = candidate
                break
        if videos_root is None:
            videos_root = root

    out_root = Path(args.out_root)
    ensure_dirs(out_root, ["train", "val", "test"])

    crop_cfg = CropConfig(pad=args.pad, min_side=args.min_side)

    rows = list(iter_labels_csv(labels_csv_path))
    if not rows:
        print("No rows found in labels CSV.")
        return

    saved, skipped = 0, 0

    # Group by video to avoid reopening many times; but keep simple iteration
    for i, row in enumerate(tqdm(rows, desc="Converting DAiSEE")):
        # Determine target split
        sp = guess_split(row, args.split)
        if sp not in {"train", "val", "test"}:
            sp = args.split

        # Resolve video path
        vid_path = resolve_video_path(row, root, videos_root)
        if vid_path is None or not vid_path.exists():
            skipped += 1
            continue

        # Compute class label
        label = label_from_daisee_row(row, args.conf_thr, args.frus_thr, args.bor_thr, args.eng_att, args.eng_dist)
        if label not in TARGET_CLASSES:
            skipped += 1
            continue

        # Frame index handling
        frame_idx: Optional[int] = None
        if args.frame_col in row and row[args.frame_col] not in (None, ""):
            try:
                frame_idx = int(float(row[args.frame_col]))
            except Exception:
                frame_idx = None

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            skipped += 1
            continue

        try:
            if frame_idx is None:
                # Extract a frame every N frames; here, pick the middle frame of first second as a proxy
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_idx = int(max(0, fps // 2))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                # fallback to first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = cap.read()
                if not ok:
                    skipped += 1
                    continue

            out_name = f"{i:08d}.jpg"
            out_path = out_root / sp / label / out_name
            if extract_and_save_frame(bgr, out_path, args.img_size, args.use_mediapipe, crop_cfg, args.blur_thr):
                saved += 1
            else:
                skipped += 1
        finally:
            cap.release()

    print(f"Done. Saved: {saved}, Skipped: {skipped}. Output root: {out_root}")


if __name__ == "__main__":
    main()
