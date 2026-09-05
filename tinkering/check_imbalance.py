import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from pathlib import Path
import pandas as pd

# Path to your data directory
data_dir = r"C:\\tinkering\\tinkering\\data\\processed_engagement_from_affectnet_folder"  # Update this path if needed

# Class names from your training script
class_names = ["attentive", "distracted", "confused", "disengaged"]

def analyze_class_distribution(data_dir):
    """Analyze and visualize class distribution in the dataset."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Check for train/val/test structure
    splits = ['train', 'val', 'test']
    class_counts = {split: Counter() for split in splits}
    
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found in {data_dir}")
            continue
            
        # Get all class directories
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                # Count images in this class
                count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png'))) + len(list(class_dir.glob('*.jpeg')))
                class_name = class_dir.name
                class_counts[split][class_name] += count
    
    # Check if any images were found
    total_images = sum(sum(counts.values()) for counts in class_counts.values())
    if total_images == 0:
        print(f"No images found in {data_dir}")
        return None
        
    print(f"Found {total_images} images across all splits")
    
    # Print and plot class distribution
    all_classes = set()
    for split in splits:
        all_classes.update(class_counts[split].keys())
    all_classes = sorted(list(all_classes))
    
    # Create a DataFrame for better visualization
    df_data = []
    for split in splits:
        for class_name in all_classes:
            count = class_counts[split].get(class_name, 0)
            df_data.append({'Split': split, 'Class': class_name, 'Count': count})
    
    df = pd.DataFrame(df_data)
    
    # Print summary
    print("\nClass Distribution Summary:")
    print("-" * 50)
    for split in splits:
        total = sum(class_counts[split].values())
        print(f"\n{split.upper()} SET ({total} total samples):")
        for class_name in all_classes:
            count = class_counts[split].get(class_name, 0)
            if total > 0:
                percentage = (count / total) * 100
                print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
            else:
                print(f"  {class_name}: {count} samples")
    
    # Plot class distribution
    plt.figure(figsize=(14, 6))
    
    # Plot each split as a separate bar group
    bar_width = 0.25
    index = np.arange(len(all_classes))
    
    for i, split in enumerate(splits):
        counts = [class_counts[split].get(c, 0) for c in all_classes]
        plt.bar(index + i*bar_width, counts, bar_width, label=split.capitalize())
    
    plt.title('Class Distribution by Split', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(index + bar_width, all_classes, rotation=45, ha='right')
    plt.legend()
    
    # Add value labels on top of bars
    for i, split in enumerate(splits):
        for j, class_name in enumerate(all_classes):
            count = class_counts[split].get(class_name, 0)
            if count > 0:  # Only label non-zero counts
                total = sum(class_counts[split].values())
                percentage = (count / total) * 100 if total > 0 else 0
                plt.text(
                    j + i*bar_width, 
                    count + 5,  # Offset above the bar
                    f'{count}\n({percentage:.1f}%)',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("\nSaved class distribution plot to 'class_distribution.png'")
    
    # Calculate and print class weights for training
    if 'train' in class_counts:
        train_counts = class_counts['train']
        if train_counts:
            print("\nClass Weights (for loss function):")
            total = sum(train_counts.values())
            for class_name in all_classes:
                count = train_counts.get(class_name, 0)
                if count > 0:
                    weight = total / (len(train_counts) * count)
                    print(f"  {class_name}: {weight:.4f}")
    
    return df

if __name__ == "__main__":
    print(f"Analyzing dataset at: {data_dir}")
    df = analyze_class_distribution(data_dir)
    
    # Save detailed distribution to CSV
    if df is not None and not df.empty:
        df_pivot = df.pivot(index='Class', columns='Split', values='Count').fillna(0)
        df_pivot.to_csv('class_distribution.csv')
        print("\nSaved detailed class distribution to 'class_distribution.csv'")
    
    print("\nAnalysis complete. Check 'class_distribution.png' for visualization.")
