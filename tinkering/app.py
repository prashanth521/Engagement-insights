import argparse
import time
from typing import Optional
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

from engagement import EngagementEstimator, EngagementConfig, extract_metrics, EngagementState
from infer import load_model
import torchvision.transforms as T
from PIL import Image
import torch
import torchvision

mp_face_mesh = mp.solutions.face_mesh


def draw_overlay(frame, metrics, state: Optional[EngagementState], stats, model_pred=None, class_probs=None, class_names=None):
    h, w = frame.shape[:2]
    color = (0, 200, 0)
    if state == EngagementState.DISTRACTED:
        color = (0, 165, 255)
    elif state == EngagementState.CONFUSED:
        color = (255, 0, 0)
    elif state == EngagementState.DISENGAGED:
        color = (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (520, 220), (0, 0, 0), -1)
    y = 35
    if state is not None:
        cv2.putText(frame, f"Heuristic: {state.value}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28
    if model_pred is not None:
        m_label, m_conf = model_pred
        cv2.putText(frame, f"Model: {m_label} ({m_conf:.2f})", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
        y += 24
    if class_probs is not None and class_names is not None:
        # Show per-class probabilities (top-4)
        try:
            pairs = list(zip(class_names, class_probs))
            # Sort by probability desc
            pairs.sort(key=lambda p: p[1], reverse=True)
            for name, p in pairs[:4]:
                cv2.putText(frame, f"{name}: {p:.2f}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
                y += 20
        except Exception:
            pass
    if metrics is not None:
        cv2.putText(frame, f"EAR: {metrics.avg_ear:.2f}  MAR: {metrics.mar:.2f}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24
        cv2.putText(frame, f"GazeProxy: {metrics.gaze_forward_proxy:.2f}  Tilt: {metrics.head_tilt_deg:.1f}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 24
    if stats:
        cv2.putText(frame, f"fwd%: {stats.get('pct_forward',0)*100:.0f}  yawn%: {stats.get('pct_yawn',0)*100:.0f}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="", help="Path to video file or rtsp/http url. If empty, use webcam 0")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index if --video not provided")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint to enable ML model predictions")
    parser.add_argument("--model-only", action="store_true", help="Disable heuristic; show only model predictions")
    parser.add_argument("--smooth-window", type=int, default=5, help="Temporal smoothing window for model probs (1=off)")
    # Heuristic tuning
    parser.add_argument("--ear-closed", type=float, default=None, help="Override ear_closed_threshold (default from EngagementConfig)")
    parser.add_argument("--ear-maybe", type=float, default=None, help="Override ear_maybe_closed_threshold")
    parser.add_argument("--mar-yawn", type=float, default=None, help="Override mar_yawn_threshold")
    parser.add_argument("--gaze-min", type=float, default=None, help="Override gaze_forward_min")
    parser.add_argument("--tilt-max", type=float, default=None, help="Override head_tilt_abs_max_deg")
    parser.add_argument("--away-seconds", type=float, default=None, help="Override away_duration_seconds")
    args = parser.parse_args()


    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    # Build EngagementConfig with possible overrides
    estimator = None
    cfg = None
    if not args.model_only:
        cfg = EngagementConfig()
        if args.ear_closed is not None:
            cfg.ear_closed_threshold = float(args.ear_closed)
        if args.ear_maybe is not None:
            cfg.ear_maybe_closed_threshold = float(args.ear_maybe)
        if args.mar_yawn is not None:
            cfg.mar_yawn_threshold = float(args.mar_yawn)
        if args.gaze_min is not None:
            cfg.gaze_forward_min = float(args.gaze_min)
        if args.tilt_max is not None:
            cfg.head_tilt_abs_max_deg = float(args.tilt_max)
        if args.away_seconds is not None:
            cfg.away_duration_seconds = float(args.away_seconds)
        estimator = EngagementEstimator(cfg)

    model = None
    classes = None
    tf = None
    
    # Enhanced device detection with multiple backends
    device = None
    
    # Check for DirectML
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"Using DirectML: {torch_directml.device_name(0) if hasattr(torch_directml, 'device_name') else 'DirectML device'}")
    except ImportError:
        pass
    
    # If DirectML not available, try other backends
    if device is None:
        # Check for Intel XPU (for Intel Arc GPUs)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device('xpu')
            print(f"Using Intel XPU: {torch.xpu.get_device_name(0) if torch.xpu.device_count() > 0 else 'XPU device'}")
        # Fall back to CUDA if available
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        # Default to CPU
        else:
            device = torch.device('cpu')
            print("No GPU acceleration available. Using CPU.")
    
    # Print environment information
    print(f"PyTorch version: {torch.__version__}")
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"Intel Extension for PyTorch version: {torch.xpu.get_device_properties(0).driver_version if torch.xpu.device_count() > 0 else 'N/A'}")
    print(f"Running on device: {device}")
    print(f"Device count: {torch_directml.device_count() if hasattr(torch, 'directml') else torch.xpu.device_count() if hasattr(torch, 'xpu') else torch.cuda.device_count() if torch.cuda.is_available() else 1}")
    print(f"Device count: {torch.xpu.device_count() if hasattr(torch, 'xpu') else torch.cuda.device_count() if torch.cuda.is_available() else 1}")
    # Resolve checkpoint path: prefer --ckpt, otherwise auto-pick latest from tinkering/checkpoints
    ckpt_path: Optional[str] = None
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            print(f"Error: checkpoint not found: {args.ckpt}")
            print("Please provide a valid path to a .pt file saved by tools/train_engagement_finetune.py")
            return
        ckpt_path = args.ckpt
    else:
        # Auto-discover in default checkpoints directory
        default_dir = Path(__file__).resolve().parent / "checkpoints"
        if default_dir.exists():
            # Look for both .pt and .pth
            candidates = list(default_dir.glob("*.pt")) + list(default_dir.glob("*.pth"))
            pts = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                ckpt_path = str(pts[0])
                print(f"Auto-selected checkpoint: {ckpt_path}")
    if ckpt_path:
        print(f"\n{'='*50}")
        print(f"Loading model from {ckpt_path}")
        print(f"Target device: {device}")
        print(f"{'='*50}\n")
        
        try:
            print(f"\n{'='*50}")
            print("LOADING MODEL...")
            
            # Initialize default values
            classes = ['attentive', 'confused', 'disengaged', 'distracted']
            tf = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            try:
                # Load the checkpoint
                checkpoint = torch.load(ckpt_path, map_location=device)
                
                # Create the model with the correct architecture
                from model import create_model
                model = create_model(num_classes=len(classes), pretrained=False)
                
                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    # Handle state dict format
                    model.load_state_dict(checkpoint['state_dict'])
                    if 'classes' in checkpoint:
                        classes = checkpoint['classes']
                    if 'transform' in checkpoint:
                        tf = checkpoint['transform']
                elif 'model_state_dict' in checkpoint:
                    # Handle model_state_dict format
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'classes' in checkpoint:
                        classes = checkpoint['classes']
                elif 'model' in checkpoint:
                    # Handle full model in 'model' key
                    model = checkpoint['model']
                    if 'classes' in checkpoint:
                        classes = checkpoint['classes']
                    if 'transform' in checkpoint:
                        tf = checkpoint['transform']
                else:
                    # Assume the checkpoint is the state dict directly
                    try:
                        model.load_state_dict(checkpoint)
                    except Exception as e:
                        print(f"Error loading state dict: {e}")
                        # If state dict loading fails, try loading as full model
                        model = checkpoint
                
                # Move model to device and set to eval mode
                model = model.to(device)
                model.eval()
                
                # Print model information
                print(f"\n{'='*50}")
                print("MODEL ARCHITECTURE:")
                print(model)
                
                # Verify the model can do a forward pass
                with torch.no_grad():
                    test_input = torch.randn(1, 3, 224, 224).to(device)
                    output = model(test_input)
                    print(f"\nTest forward pass output shape: {output.shape}")
                    print(f"Output values: {output}")
                    
                    # Check if output is all zeros
                    if torch.all(output == 0):
                        print("\nWARNING: Model is outputting all zeros!")
                        print("This suggests the model weights might not be properly loaded.")
                        print("Trying to fix by reinitializing the final layer...")
                        
                        # Reinitialize the final layer
                        if hasattr(model, 'backbone') and hasattr(model.backbone, 'fc'):
                            in_features = model.backbone.fc[1].in_features
                            model.backbone.fc[1] = nn.Linear(in_features, len(classes)).to(device)
                            print("Reinitialized the final layer.")
                        
                        # Test again
                        output = model(test_input)
                        print(f"New output values: {output}")
                
            except Exception as e:
                print(f"\nError loading model: {str(e)}")
                print("Falling back to a fresh model with random weights...")
                from model import create_model
                model = create_model(num_classes=len(classes), pretrained=False).to(device)
                model.eval()
            
            # Print transform information
            print("\n" + "="*50)
            print("TRANSFORM COMPOSITION:")
            print(tf)
            print(f"\nModel moved to {device}, mode: {'training' if model.training else 'evaluation'}")
            
            # Verify model parameters
            print(f"\n{'='*50}")
            print("MODEL PARAMETERS:")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Check first and last layer weights
            print("\nMODEL WEIGHTS CHECK:")
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) > 1:  # Only check weight matrices
                    print(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                if name in ['fc.weight', 'classifier.weight', 'backbone.fc.1.weight']:  # Common names for final layer
                    final_weights = param.data
                    print(f"\nFinal layer weights (first 5x5):")
                    print(final_weights[:5, :5].cpu().numpy())
            
            # Run a test inference with debug info
            print("\n" + "="*50)
            print("TEST INFERENCE:")
            with torch.no_grad():
                # Create a test input (normalized random noise)
                test_input = torch.randn(1, 3, 224, 224).to(device) * 0.1  # Small random values
                
                # Apply the same transform used for real images
                if hasattr(tf, 'transforms') and len(tf.transforms) > 0:
                    # If transform is a composition, apply normalization if present
                    for t in tf.transforms:
                        if isinstance(t, torchvision.transforms.Normalize):
                            test_input = t(test_input)
                            print("Applied normalization from transform")
                            break
                
                print(f"Test input - shape: {test_input.shape}, "
                      f"range: [{test_input.min().item():.4f}, {test_input.max().item():.4f}], "
                      f"mean: {test_input.mean().item():.4f}, std: {test_input.std().item():.4f}")
                
                # Run model
                output = model(test_input)
                print(f"\nModel output - shape: {output.shape}")
                print(f"Raw logits: {output.data.cpu().numpy()}")
                
                # Check for NaN/Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print("WARNING: Model output contains NaN or Inf values!")
                
                # Calculate probabilities
                probs = torch.softmax(output, dim=1)
                print("\nProbabilities (softmax):")
                for i, p in enumerate(probs[0]):
                    print(f"  {classes[i] if i < len(classes) else f'Class {i}'}: {p.item():.4f}")
                
                # Check if probabilities are uniform
                uniform_prob = 1.0 / probs.shape[1]
                if torch.allclose(probs, torch.ones_like(probs) * uniform_prob, atol=1e-2):
                    print("\nWARNING: Model is outputting uniform probabilities!")
                else:
                    print("\nModel is producing non-uniform probabilities as expected.")
            
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"\n{'!'*50}")
            print(f"Error loading model: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Make sure the checkpoint file is not corrupted")
            print("2. Verify that the model architecture matches the checkpoint")
            print("3. Check if you have the correct version of PyTorch installed")
            if hasattr(torch, 'xpu'):
                print("4. Ensure Intel Extension for PyTorch is properly installed")
            print(f"{'!'*50}\n")
            return
    elif args.model_only:
        print("Error: --model-only requires --ckpt pointing to a valid checkpoint file")
        return
    probs_history = deque(maxlen=max(1, int(args.smooth_window))) if (args.ckpt and args.smooth_window and args.smooth_window > 1) else deque(maxlen=1)

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
            state: Optional[EngagementState] = None if args.model_only else EngagementState.DISTRACTED
            model_pred = None
            class_probs = None
            class_names = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                pts = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark], dtype=np.float32)
                if not args.model_only:
                    metrics = extract_metrics(pts, w, h)
                    if metrics is not None and estimator is not None:
                        now = time.time()
                        state = estimator.update(metrics, now)

                # Compute face bbox from landmarks (used for drawing and optional model inference)
                xs = (pts[:, 0] * w).astype(np.int32)
                ys = (pts[:, 1] * h).astype(np.int32)
                x1, y1 = max(xs.min() - 20, 0), max(ys.min() - 20, 0)
                x2, y2 = min(xs.max() + 20, w - 1), min(ys.max() + 20, h - 1)

                if model is not None:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        with torch.no_grad():
                            try:
                                # Convert image to tensor and move to device
                                x = tf(img).unsqueeze(0)
                                
                                # Special handling for Intel XPU
                                if hasattr(torch, 'xpu') and device.type == 'xpu':
                                    x = x.xpu()
                                else:
                                    x = x.to(device)
                                
                                try:
                                    # Run inference
                                    with torch.no_grad():
                                        logits = model(x)
                                        
                                        # Calculate probabilities
                                        probs_softmax = torch.softmax(logits, dim=1)
                                        
                                        # Move results to CPU and convert to list
                                        if hasattr(torch, 'xpu') and device.type == 'xpu':
                                            probs = probs_softmax.cpu().numpy()[0].tolist()
                                        else:
                                            probs = probs_softmax[0].detach().cpu().numpy().tolist()
                                            
                                except Exception as e:
                                    print(f"Error during inference: {str(e)}")
                                    probs = [1.0/len(classes)] * len(classes)  # Uniform probabilities as fallback
                                
                                # Update smoothing history
                                probs_history.append(probs)
                                
                                # Compute smoothed probabilities
                                if len(probs_history) > 1:
                                    smoothed = np.mean(np.array(probs_history, dtype=np.float32), axis=0).tolist()
                                else:
                                    smoothed = probs
                                
                                # Determine top class from smoothed probs
                                idx = int(np.argmax(smoothed))
                                conf = float(smoothed[idx])
                                model_pred = (classes[idx], conf)
                                class_probs = smoothed
                                class_names = classes
                                
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower():
                                    print("\n" + "!"*50)
                                    print("Out of memory error! Try reducing the batch size or image resolution.")
                                    print("Current device:", device)
                                    print("Allocated memory:", 
                                          torch.xpu.memory_allocated(device)/1e6 if hasattr(torch, 'xpu') else 
                                          torch.cuda.memory_allocated(device)/1e6 if torch.cuda.is_available() else 0, "MB")
                                    print("Max memory allocated:", 
                                          torch.xpu.max_memory_allocated(device)/1e6 if hasattr(torch, 'xpu') else 
                                          torch.cuda.max_memory_allocated(device)/1e6 if torch.cuda.is_available() else 0, "MB")
                                    print("!"*50 + "\n")
                                raise

                # draw face bbox
                if results.multi_face_landmarks:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 1)

                # landmarks (subset)
                for idx in [33, 133, 362, 263, 159, 145, 386, 374, 13, 14, 61, 291, 1, 152]:
                    x, y = int(pts[idx, 0] * w), int(pts[idx, 1] * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            stats = estimator.stats() if (estimator is not None) else {}
            draw_overlay(frame, metrics, state, stats, model_pred, class_probs, class_names)
            cv2.imshow("Engagement Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 