import argparse
from pathlib import Path
from collections import Counter
import os

def analyze_dataset(data_dir: str):
    """Analyze dataset balance and show class distribution."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    # Count files in each class
    class_counts = {}
    total_files = 0
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            file_count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg")))
            class_counts[class_name] = file_count
            total_files += file_count
    
    print(f"\nDataset Analysis: {data_dir}")
    print("=" * 50)
    print(f"Total files: {total_files}")
    print("\nClass distribution:")
    print("-" * 30)
    
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"{class_name:12}: {count:6} files ({percentage:5.1f}%)")
    
    # Check for imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nImbalance analysis:")
        print(f"Max class: {max_count} files")
        print(f"Min class: {min_count} files")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 3:
            print("⚠️  WARNING: Severe class imbalance detected!")
            print("   Consider:")
            print("   - Data augmentation for minority classes")
            print("   - Class weights in training")
            print("   - Collecting more data for minority classes")
        elif imbalance_ratio > 2:
            print("⚠️  Moderate class imbalance detected")
        else:
            print("✅ Dataset appears balanced")
    
    # Check for missing classes
    expected_classes = {"attentive", "distracted", "confused", "disengaged"}
    missing_classes = expected_classes - set(class_counts.keys())
    if missing_classes:
        print(f"\n⚠️  Missing classes: {missing_classes}")
    
    extra_classes = set(class_counts.keys()) - expected_classes
    if extra_classes:
        print(f"\nℹ️  Extra classes found: {extra_classes}")

def main():
    parser = argparse.ArgumentParser(description="Analyze engagement dataset balance")
    parser.add_argument("data_dir", help="Path to dataset directory (ImageFolder structure)")
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir)

if __name__ == "__main__":
    main()
