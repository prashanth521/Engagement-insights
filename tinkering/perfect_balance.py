import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def perfect_balance(src_dir, dest_dir, test_size=0.15, val_size=0.15, min_samples=None):
    """
    Create perfectly balanced splits with equal number of samples per class.
    Uses the class with the fewest samples as the reference.
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    
    # Find the class with the fewest samples
    classes = [d.name for d in (src_dir / 'train').iterdir() if d.is_dir()]
    class_counts = {}
    
    print("Analyzing class sizes...")
    for class_name in classes:
        count = len(list((src_dir / 'train' / class_name).glob('*.*')))
        class_counts[class_name] = count
        print(f"  {class_name}: {count} samples")
    
    # Use the smallest class size as reference (or min_samples if provided)
    min_count = min(class_counts.values()) if min_samples is None else min_samples
    samples_per_class = {
        'train': int(min_count * (1 - test_size - val_size)),
        'val': int(min_count * val_size),
        'test': int(min_count * test_size)
    }
    
    print(f"\nUsing {min_count} samples per class")
    print(f"Samples per split: {samples_per_class}")
    
    # Create destination directories
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            (dest_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Get all images for this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend((src_dir / 'train' / class_name).glob(ext))
            image_files.extend((src_dir / 'val' / class_name).glob(ext))
            image_files.extend((src_dir / 'test' / class_name).glob(ext))
        
        # Shuffle and limit to min_count
        random.seed(42)
        random.shuffle(image_files)
        image_files = image_files[:min_count]
        
        # Split into train, val, test
        train_end = samples_per_class['train']
        val_end = train_end + samples_per_class['val']
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:val_end + samples_per_class['test']]
        }
        
        # Copy files
        for split, files in splits.items():
            print(f"  {split}: {len(files)} samples")
            for src_file in tqdm(files, desc=f"    Copying to {split}", leave=False):
                dst_file = dest_dir / split / class_name / src_file.name
                if not dst_file.exists():  # Skip if already exists
                    shutil.copy2(src_file, dst_file)

def main():
    # Configuration
    src_data_dir = r"C:\tinkering\tinkering\data\processed_engagement_from_affectnet_folder"
    dest_data_dir = r"C:\tinkering\tinkering\data\perfectly_balanced_data"
    
    print(f"Source directory: {src_data_dir}")
    print(f"Destination directory: {dest_data_dir}")
    
    # Perfectly balance the data
    perfect_balance(
        src_dir=src_data_dir,
        dest_dir=dest_data_dir,
        test_size=0.15,  # 15% for test
        val_size=0.15,   # 15% for validation
        min_samples=1000  # Set this to the desired number of samples per class
    )
    
    print("\nPerfect balancing complete!")
    print(f"Balanced dataset saved to: {dest_data_dir}")

if __name__ == "__main__":
    main()
