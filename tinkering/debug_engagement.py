import argparse
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from engagement import EngagementEstimator, EngagementConfig, extract_metrics, EngagementState

mp_face_mesh = mp.solutions.face_mesh


def debug_overlay(frame, metrics, state: EngagementState, config: EngagementConfig):
    h, w = frame.shape[:2]
    
    # Show raw metrics and thresholds
    cv2.rectangle(frame, (10, 10), (600, 200), (0, 0, 0), -1)
    y = 35
    
    # State
    color = (0, 200, 0)
    if state == EngagementState.DISTRACTED:
        color = (0, 165, 255)
    elif state == EngagementState.CONFUSED:
        color = (255, 0, 0)
    elif state == EngagementState.DISENGAGED:
        color = (0, 0, 255)
    
    cv2.putText(frame, f"State: {state.value}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y += 30
    
    if metrics is not None:
        # EAR analysis
        ear_status = "CLOSED" if metrics.avg_ear < config.ear_closed_threshold else "OPEN"
        ear_color = (0, 0, 255) if metrics.avg_ear < config.ear_closed_threshold else (0, 255, 0)
        cv2.putText(frame, f"EAR: {metrics.avg_ear:.3f} (thresh: {config.ear_closed_threshold:.3f}) - {ear_status}", 
                   (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 1)
        y += 25
        
        # MAR analysis
        mar_status = "YAWNING" if metrics.mar > config.mar_yawn_threshold else "NORMAL"
        mar_color = (0, 0, 255) if metrics.mar > config.mar_yawn_threshold else (0, 255, 0)
        cv2.putText(frame, f"MAR: {metrics.mar:.3f} (thresh: {config.mar_yawn_threshold:.3f}) - {mar_status}", 
                   (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 1)
        y += 25
        
        # Gaze analysis
        gaze_status = "FORWARD" if metrics.gaze_forward_proxy >= config.gaze_forward_min else "AWAY"
        gaze_color = (0, 255, 0) if metrics.gaze_forward_proxy >= config.gaze_forward_min else (0, 165, 255)
        cv2.putText(frame, f"Gaze: {metrics.gaze_forward_proxy:.3f} (thresh: {config.gaze_forward_min:.3f}) - {gaze_status}", 
                   (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_color, 1)
        y += 25
        
        # Head tilt analysis
        tilt_status = "NORMAL" if abs(metrics.head_tilt_deg) <= config.head_tilt_abs_max_deg else "TILTED"
        tilt_color = (0, 255, 0) if abs(metrics.head_tilt_deg) <= config.head_tilt_abs_max_deg else (0, 165, 255)
        cv2.putText(frame, f"Tilt: {metrics.head_tilt_deg:.1f}° (max: {config.head_tilt_abs_max_deg:.1f}°) - {tilt_status}", 
                   (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_color, 1)
        y += 25
        
        # Decision logic
        is_eyes_closed = metrics.avg_ear < config.ear_closed_threshold
        is_yawning = metrics.mar > config.mar_yawn_threshold
        is_forward = metrics.gaze_forward_proxy >= config.gaze_forward_min and abs(metrics.head_tilt_deg) <= config.head_tilt_abs_max_deg
        
        cv2.putText(frame, f"Eyes closed: {is_eyes_closed} | Yawning: {is_yawning} | Forward: {is_forward}", 
                   (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if isinstance(args.camera, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    # Use more lenient thresholds for debugging
    config = EngagementConfig(
        ear_closed_threshold=0.15,  # More lenient
        ear_maybe_closed_threshold=0.20,  # More lenient
        mar_yawn_threshold=0.7,  # More lenient
        gaze_forward_min=1.5,  # More lenient
        head_tilt_abs_max_deg=25.0,  # More lenient
        away_duration_seconds=8.0,  # More lenient
    )
    
    estimator = EngagementEstimator(config)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            h, w = frame.shape[:2]
            metrics = None
            state = EngagementState.DISTRACTED

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                pts = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark], dtype=np.float32)
                metrics = extract_metrics(pts, w, h)
                if metrics is not None:
                    now = time.time()
                    state = estimator.update(metrics, now)

                # Draw face bbox
                xs = (pts[:, 0] * w).astype(np.int32)
                ys = (pts[:, 1] * h).astype(np.int32)
                x1, y1 = max(xs.min() - 20, 0), max(ys.min() - 20, 0)
                x2, y2 = min(xs.max() + 20, w - 1), min(ys.max() + 20, h - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 1)

                # Draw key landmarks
                for idx in [33, 133, 362, 263, 159, 145, 386, 374, 13, 14, 61, 291, 1, 152]:
                    x, y = int(pts[idx, 0] * w), int(pts[idx, 1] * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            debug_overlay(frame, metrics, state, config)
            cv2.imshow("Engagement Debug", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
