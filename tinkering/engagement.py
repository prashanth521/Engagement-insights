import enum
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

# MediaPipe FaceMesh landmark indices used
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt
# We use a subset for eyes and mouth metrics
LEFT_EYE_LANDMARKS = {
    "upper": 159,  # top eyelid
    "lower": 145,  # bottom eyelid
    "left": 33,    # outer corner
    "right": 133,  # inner corner
}
RIGHT_EYE_LANDMARKS = {
    "upper": 386,
    "lower": 374,
    "left": 362,
    "right": 263,
}
MOUTH_LANDMARKS = {
    "upper_inner": 13,  # upper lip inner
    "lower_inner": 14,  # lower lip inner
    "left": 61,
    "right": 291,
}
NOSE_TIP = 1
FOREHEAD_REF = 10  # approximate top
CHIN = 152


class EngagementState(enum.Enum):
    ATTENTIVE = "Attentive/Interested"
    DISTRACTED = "Distracted/Bored"
    CONFUSED = "Confused"
    DISENGAGED = "Disengaged/Sleeping"


@dataclass
class FacialMetrics:
    left_ear: float
    right_ear: float
    mar: float
    gaze_forward_proxy: float
    head_tilt_deg: float

    @property
    def avg_ear(self) -> float:
        return (self.left_ear + self.right_ear) / 2.0


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_ear(upper: Tuple[float, float], lower: Tuple[float, float], left: Tuple[float, float], right: Tuple[float, float]) -> float:
    vertical = _euclidean(upper, lower)
    horizontal = _euclidean(left, right)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


def compute_mar(upper_inner: Tuple[float, float], lower_inner: Tuple[float, float], left: Tuple[float, float], right: Tuple[float, float]) -> float:
    vertical = _euclidean(upper_inner, lower_inner)
    horizontal = _euclidean(left, right)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


def compute_head_tilt(nose: Tuple[float, float], chin: Tuple[float, float]) -> float:
    # Angle relative to vertical axis (in degrees)
    dy = chin[1] - nose[1]
    dx = chin[0] - nose[0]
    angle_rad = np.arctan2(dx, dy)  # swap to measure deviation from vertical
    return float(np.degrees(angle_rad))


def compute_gaze_forward_proxy(left_eye_left: Tuple[float, float], left_eye_right: Tuple[float, float], right_eye_left: Tuple[float, float], right_eye_right: Tuple[float, float]) -> float:
    # Proxy: ratio of inter-eye horizontal distance to vertical spread; higher suggests frontal view
    inter_left = _euclidean(left_eye_left, left_eye_right)
    inter_right = _euclidean(right_eye_left, right_eye_right)
    inter_eye = _euclidean(left_eye_right, right_eye_left)
    denom = inter_left + inter_right
    if denom <= 1e-6:
        return 0.0
    return float(inter_eye / denom)


def extract_metrics(landmarks: np.ndarray, image_width: int, image_height: int) -> Optional[FacialMetrics]:
    try:
        # landmarks: (468, 2) normalized to [0,1] coordinates
        def denorm(idx: int) -> Tuple[float, float]:
            x = float(landmarks[idx, 0] * image_width)
            y = float(landmarks[idx, 1] * image_height)
            return (x, y)

        le = LEFT_EYE_LANDMARKS
        re = RIGHT_EYE_LANDMARKS
        mo = MOUTH_LANDMARKS

        left_ear = compute_ear(denorm(le["upper"]), denorm(le["lower"]), denorm(le["left"]), denorm(le["right"]))
        right_ear = compute_ear(denorm(re["upper"]), denorm(re["lower"]), denorm(re["left"]), denorm(re["right"]))
        mar = compute_mar(denorm(mo["upper_inner"]), denorm(mo["lower_inner"]), denorm(mo["left"]), denorm(mo["right"]))
        head_tilt_deg = compute_head_tilt(denorm(NOSE_TIP), denorm(CHIN))
        gaze_proxy = compute_gaze_forward_proxy(denorm(le["left"]), denorm(le["right"]), denorm(re["left"]), denorm(re["right"]))

        return FacialMetrics(
            left_ear=left_ear,
            right_ear=right_ear,
            mar=mar,
            gaze_forward_proxy=gaze_proxy,
            head_tilt_deg=head_tilt_deg,
        )
    except Exception:
        return None


@dataclass
class EngagementConfig:
    ear_closed_threshold: float = 0.12  # More conservative to reduce false 'eyes closed'
    ear_maybe_closed_threshold: float = 0.20  # More lenient (was 0.23)
    mar_yawn_threshold: float = 0.8  # Require wider mouth to mark yawns
    gaze_forward_min: float = 1.3  # Slightly easier to be considered forward
    head_tilt_abs_max_deg: float = 25.0  # More lenient (was 20.0)
    away_duration_seconds: float = 8.0  # More lenient (was 5.0)
    window_seconds: float = 30.0
    fps_assumption: float = 15.0


class EngagementEstimator:
    def __init__(self, config: Optional[EngagementConfig] = None) -> None:
        self.config = config or EngagementConfig()
        self.history: Deque[Dict] = deque(maxlen=int(self.config.window_seconds * self.config.fps_assumption))
        self.last_forward_ts: Optional[float] = None
        self._eyes_closed_consec: int = 0

    def update(self, metrics: FacialMetrics, timestamp: float) -> EngagementState:
        cfg = self.config
        is_eyes_closed = metrics.avg_ear < cfg.ear_closed_threshold
        is_yawning = metrics.mar > cfg.mar_yawn_threshold
        is_forward = metrics.gaze_forward_proxy >= cfg.gaze_forward_min and abs(metrics.head_tilt_deg) <= cfg.head_tilt_abs_max_deg

        if is_forward:
            self.last_forward_ts = timestamp

        self.history.append({
            "ts": timestamp,
            "avg_ear": metrics.avg_ear,
            "mar": metrics.mar,
            "gaze_forward": is_forward,
            "head_tilt_deg": metrics.head_tilt_deg,
            "yawn": is_yawning,
            "eyes_closed": is_eyes_closed,
        })

        # Rule-based classification
        # High priority: disengaged
        # Only apply away-duration rule after we've seen the user facing forward at least once.
        time_since_forward = None
        if self.last_forward_ts is not None:
            time_since_forward = max(0.0, timestamp - self.last_forward_ts)

        # Require a few consecutive frames of eyes-closed to avoid flicker
        if is_eyes_closed:
            self._eyes_closed_consec += 1
        else:
            self._eyes_closed_consec = 0

        if self._eyes_closed_consec >= 5 or is_yawning:
            return EngagementState.DISENGAGED
        if time_since_forward is not None and time_since_forward > cfg.away_duration_seconds:
            return EngagementState.DISENGAGED

        # Confused: forward but frequent mouth small opens or head tilt high (but not extreme)
        # Simple heuristic via intermediate thresholds
        maybe_tired = metrics.avg_ear < cfg.ear_maybe_closed_threshold
        if is_forward and (maybe_tired or abs(metrics.head_tilt_deg) > (0.5 * cfg.head_tilt_abs_max_deg)):
            return EngagementState.CONFUSED

        if is_forward:
            return EngagementState.ATTENTIVE

        return EngagementState.DISTRACTED

    def stats(self) -> Dict[str, float]:
        if not self.history:
            return {}
        arr = list(self.history)
        num = len(arr)
        return {
            "avg_ear": float(np.mean([x["avg_ear"] for x in arr])),
            "avg_mar": float(np.mean([x["mar"] for x in arr])),
            "pct_forward": float(np.mean([1.0 if x["gaze_forward"] else 0.0 for x in arr])),
            "pct_yawn": float(np.mean([1.0 if x["yawn"] else 0.0 for x in arr])),
            "pct_eyes_closed": float(np.mean([1.0 if x["eyes_closed"] else 0.0 for x in arr])),
        } 