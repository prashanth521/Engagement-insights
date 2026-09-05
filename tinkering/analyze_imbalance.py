import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# Import our DAiSEE dataset class
from prepare_daisee import DAiSEEDataset

def analyze_class_distribution(data_dir='data/daisee'):
    """Analyze and visualize class distribution in the dataset."""
    # Create datasets without transformations for analysis
    datasets = {
        'train': DAiSEEDataset(data_dir, split='train', transform=None),
        'val': DAiSEEDataset(data_dir, split='val', transform=None),
        'test': DAiSEEDataset(data_dir, split='test', transform=None)
    }
    
    # Collect class distributions
    distributions = {}
    
    for split, dataset in datasets.items():
        print(f"\nAnalyzing {split} set...")
        
        # Get all labels
        labels = [sample['label'].item() for sample in dataset.samples]
        
        # Count occurrences of each class
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        print(f"Total samples: {total_samples}")
        print("Class distribution:")
        for class_idx in sorted(class_counts.keys()):
            count = class_counts[class_idx]
            percentage = (count / total_samples) * 100
            print(f"  Class {class_idx}: {count} samples ({percentage:.2f}%)")
        
        # Calculate class weights for loss function
        class_weights = {}
        for class_idx in sorted(class_counts.keys()):
            # Inverse of class frequency (more weight to minority classes)
            class_weights[class_idx] = 1.0 / class_counts[class_idx]
        
        # Normalize weights
        sum_weights = sum(class_weights.values())
        class_weights = {k: v/sum_weights for k, v in class_weights.items()}
        
        print("\nClass weights (for loss function):")
        for class_idx, weight in sorted(class_weights.items()):
            print(f"  Class {class_idx}: {weight:.6f}")
        
        distributions[split] = {
            'counts': class_counts,
            'weights': class_weights,
            'total': total_samples
        }
    
    # Plot class distributions
    plot_class_distributions(distributions)
    
    return distributions

def plot_class_distributions(distributions):
    """Plot class distributions for all splits."""
    class_names = ['Disengaged', 'Barely-Engaged', 'Engaged', 'Highly-Engaged']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Class Distribution Across Dataset Splits', fontsize=14)
    
    for ax, (split, data) in zip(axes, distributions.items()):
        counts = data['counts']
        total = data['total']
        
        # Get counts for all classes (some might be missing in certain splits)
        class_indices = sorted(counts.keys())
        class_counts = [counts.get(i, 0) for i in range(4)]
        percentages = [(count / total) * 100 for count in class_counts]
        
        bars = ax.bar(class_indices, [class_counts[i] for i in class_indices], color='skyblue')
        ax.set_title(f'{split.capitalize()} Set')
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Samples')
        ax.set_xticks(class_indices)
        ax.set_xticklabels([class_names[i] for i in class_indices], rotation=45)
        
        # Add percentage labels on top of bars
        for bar, percentage in zip(bars, [percentages[i] for i in class_indices]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print("\nSaved class distribution plot to 'class_distribution.png'")

def get_class_weights(data_dir='data/daisee'):
    """Calculate class weights for the training set."""
    try:
        train_dataset = DAiSEEDataset(data_dir, split='train', transform=None)
        labels = [sample['label'].item() for sample in train_dataset.samples]
        class_counts = Counter(labels)
        
        # Calculate weights (inverse of class frequency)
        weights = {}
        total = len(labels)
        for class_idx in sorted(class_counts.keys()):
            weights[class_idx] = 1.0 / class_counts[class_idx]
        
        # Normalize weights
        sum_weights = sum(weights.values())
        weights = {k: v/sum_weights for k, v in weights.items()}
        
        # Convert to list in class order
        weight_list = [weights[i] for i in range(len(weights))]
        return weight_list
    
    except Exception as e:
        print(f"Error calculating class weights: {e}")
        # Return equal weights as fallback
        return [1.0, 1.0, 1.0, 1.0]

if __name__ == "__main__":
    distributions = analyze_class_distribution()
    print("\nClass weights for training:", get_class_weights())
