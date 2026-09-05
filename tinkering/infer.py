import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from model import create_model


def load_model(ckpt_path: str, device=None):
    """Load model checkpoint with device handling.
    
    Args:
        ckpt_path: Path to the model checkpoint
        device: The device to load the model onto. If None, will use CUDA if available, otherwise CPU.
    
    Returns:
        Tuple of (model, classes, transform)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model onto device: {device}")
    
    # Try to load with the specified device first
    try:
        # First try with the specified device
        data = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e1:
        print(f"Warning: Failed to load with {device}, trying CPU... ({e1})")
        try:
            # Fall back to CPU
            data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            print("Model loaded on CPU, will move to target device after initialization")
        except Exception as e2:
            # Try one more time with a more permissive loader
            print(f"Standard load failed, trying alternative method... ({e2})")
            try:
                data = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to load checkpoint.\n"
                    f"Error details: {e3}\n"
                    f"Possible solutions:\n"
                    f"1. Check if the checkpoint file exists and is not corrupted\n"
                    f"2. Try installing torch-directml: pip install torch-directml\n"
                    f"3. The model may be incompatible with your PyTorch version"
                )
    
    # Ensure we have the required keys
    if 'model_state' not in data or 'classes' not in data:
        raise ValueError("Invalid checkpoint format: missing required keys 'model_state' or 'classes'")
    
    classes = data.get("classes")
    img_size = data.get("img_size", 224)
    arch = data.get("arch", "resnet18")
    
    print(f"Creating model with architecture: {arch}, num_classes: {len(classes)}")
    model = create_model(num_classes=len(classes), pretrained=False, arch=arch)
    
    # Handle loading the state dict with strict=False for more flexibility
    try:
        model.load_state_dict(data["model_state"])
    except RuntimeError as e:
        print(f"Warning: Could not load state dict with strict=True: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(data["model_state"], strict=False)
    
    # Move model to the specified device
    model = model.to(device)
    model.eval()
    
    # Create transforms
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"Model loaded successfully on {device} with {len(classes)} classes")
    print(f"Input image size: {img_size}x{img_size}")
    return model, classes, tf
    return model, classes, tf


def predict_image(model, tf, image: Image.Image):
    with torch.no_grad():
        x = tf(image).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    return int(pred_idx.item()), float(conf.item()), probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("image", type=str)
    args = parser.parse_args()

    model, classes, tf = load_model(args.ckpt)
    img = Image.open(args.image).convert("RGB")
    idx, conf, probs = predict_image(model, tf, img)
    print({"pred": classes[idx], "conf": round(conf, 3), "probs": {classes[i]: round(p, 3) for i, p in enumerate(probs)}})


if __name__ == "__main__":
    main()
