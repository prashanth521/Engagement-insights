import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

TARGET_CLASSES = ["attentive", "distracted", "confused", "disengaged"]
EXPECTED_EMOTIONS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
EMOTION_ALIASES: Dict[str, str] = {"contempt": "disgust"}

# Emotion -> Engagement default mapping
DEFAULT_MAP = {
    "neutral": "attentive",
    "happy": "attentive",
    "sad": "disengaged",
    "surprise": "confused",
    "fear": "confused",
    "disgust": "distracted",
    "anger": "distracted",
}

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices (same subset as in tinkering/engagement.py)
LEFT = {"upper": 159, "lower": 145, "left": 33, "right": 133}
RIGHT = {"upper": 386, "lower": 374, "left": 362, "right": 263}
MOUTH = {"upper_inner": 13, "lower_inner": 14, "left": 61, "right": 291}
NOSE_TIP = 1
CHIN = 152


def _euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_ear(upper, lower, left, right) -> float:
    vertical = _euclidean(upper, lower)
    horizontal = _euclidean(left, right)
    return 0.0 if horizontal <= 1e-6 else vertical / horizontal


def compute_mar(upper_inner, lower_inner, left, right) -> float:
    vertical = _euclidean(upper_inner, lower_inner)
    horizontal = _euclidean(left, right)
    return 0.0 if horizontal <= 1e-6 else vertical / horizontal


def compute_head_tilt(nose, chin) -> float:
    dy = chin[1] - nose[1]
    dx = chin[0] - nose[0]
    angle_rad = np.arctan2(dx, dy)
    return float(np.degrees(angle_rad))


def compute_gaze_forward_proxy(le_left, le_right, re_left, re_right) -> float:
    inter_left = _euclidean(le_left, le_right)
    inter_right = _euclidean(re_left, re_right)
    inter_eye = _euclidean(le_right, re_left)
    denom = inter_left + inter_right
    return 0.0 if denom <= 1e-6 else float(inter_eye / denom)


def extract_metrics(landmarks: np.ndarray, w: int, h: int) -> Optional[Dict[str, float]]:
    try:
        def d(idx: int):
            x = float(landmarks[idx, 0] * w)
            y = float(landmarks[idx, 1] * h)
            return (x, y)
        left_ear = compute_ear(d(LEFT["upper"]), d(LEFT["lower"]), d(LEFT["left"]), d(LEFT["right"]))
        right_ear = compute_ear(d(RIGHT["upper"]), d(RIGHT["lower"]), d(RIGHT["left"]), d(RIGHT["right"]))
        mar = compute_mar(d(MOUTH["upper_inner"]), d(MOUTH["lower_inner"]), d(MOUTH["left"]), d(MOUTH["right"]))
        head_tilt_deg = compute_head_tilt(d(NOSE_TIP), d(CHIN))
        gaze_proxy = compute_gaze_forward_proxy(d(LEFT["left"]), d(LEFT["right"]), d(RIGHT["left"]), d(RIGHT["right"]))
        return {
            "avg_ear": (left_ear + right_ear) / 2.0,
            "mar": mar,
            "head_tilt_deg": head_tilt_deg,
            "gaze_forward_proxy": gaze_proxy,
        }
    except Exception:
        return None


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


def refine_label(default_label: str, m: Dict[str, float], th) -> str:
    if m["avg_ear"] < th["ear_closed"] or m["mar"] > th["mar_yawn"]:
        return "disengaged"
    forward = m["gaze_forward_proxy"] >= th["gaze_forward_min"] and abs(m["head_tilt_deg"]) <= th["head_tilt_max"]
    maybe_tired = m["avg_ear"] < th["ear_maybe_closed"]
    if not forward:
        if default_label == "attentive":
            return "distracted"
        return default_label
    if default_label == "attentive" and maybe_tired:
        return "confused"
    if default_label == "confused" and not maybe_tired:
        return "attentive"
    return default_label


def find_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def normalize_emotion_from_path(p: Path) -> str:
    for parent in [p.parent, *p.parents]:
        name = parent.name.strip().lower()
        if name in EXPECTED_EMOTIONS:
            return name
        if name in EMOTION_ALIASES:
            return EMOTION_ALIASES[name]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert folder-structured AffectNet to 4-class engagement ImageFolder using heuristics and optional MediaPipe proxies.")
    parser.add_argument("--images_root", type=str, required=True, help="Root containing emotion-named folders (recursively)")
    parser.add_argument("--out_root", type=str, default="tinkering/data/processed_engagement_from_affectnet_folder", help="Output root")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--use_mediapipe", action="store_true")
    parser.add_argument("--blur_thr", type=float, default=120.0)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--ear_closed", type=float, default=0.15)
    parser.add_argument("--ear_maybe_closed", type=float, default=0.20)
    parser.add_argument("--mar_yawn", type=float, default=0.7)
    parser.add_argument("--gaze_forward_min", type=float, default=1.5)
    parser.add_argument("--head_tilt_max", type=float, default=25.0)
    args = parser.parse_args()

    root = Path(args.images_root)
    out_root = Path(args.out_root)

    imgs = find_images(root)
    if not imgs:
        print("No images found under images_root.")
        return

    thresholds = {
        "ear_closed": args.ear_closed,
        "ear_maybe_closed": args.ear_maybe_closed,
        "mar_yawn": args.mar_yawn,
        "gaze_forward_min": args.gaze_forward_min,
        "head_tilt_max": args.head_tilt_max,
    }

    def which_split(i: int, n: int) -> str:
        t_end = int(n * args.train_ratio)
        v_end = t_end + int(n * args.val_ratio)
        if i < t_end:
            return "train"
        if i < v_end:
            return "val"
        return "test"

    for sp in ["train", "val", "test"]:
        for c in TARGET_CLASSES:
            (out_root / sp / c).mkdir(parents=True, exist_ok=True)

    ok, skipped = 0, 0
    n = len(imgs)
    for i, in_path in enumerate(tqdm(imgs, desc="Converting (folder->engagement ImageFolder)")):
        emo = normalize_emotion_from_path(in_path)
        if not emo:
            skipped += 1
            continue
        default_eng = DEFAULT_MAP.get(emo, "distracted")

        try:
            bgr = cv2.imread(str(in_path))
            if bgr is None:
                skipped += 1
                continue
            if is_blurry(bgr, args.blur_thr):
                skipped += 1
                continue

            if args.use_mediapipe:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                ) as fm:
                    res = fm.process(rgb)
                if not res.multi_face_landmarks:
                    skipped += 1
                    continue
                lm = res.multi_face_landmarks[0]
                pts = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
                h, w = bgr.shape[:2]
                x1, y1, x2, y2 = compute_bbox_from_landmarks(pts, w, h, pad=args.pad)
                crop = bgr[y1:y2, x1:x2]
                metrics = extract_metrics(pts, w, h)
                label = refine_label(default_eng, metrics, thresholds) if metrics is not None else default_eng
            else:
                crop = bgr
                label = default_eng

            if crop.size == 0:
                skipped += 1
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(crop_rgb).resize((args.img_size, args.img_size))
            sp = which_split(i, n)
            out_path = out_root / sp / label / f"{i:08d}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, quality=95)
            ok += 1
        except Exception:
            skipped += 1
            continue

    print(f"Done. Saved: {ok}, Skipped: {skipped}. Output: {out_root}")


if __name__ == "__main__":
    main()
