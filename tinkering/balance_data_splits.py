import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

def balance_data_splits(src_dir, dest_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Balance the dataset by creating equal class distributions in train/val/test splits.
    
    Args:
        src_dir (str): Source directory with original data (should have train/val/test subdirectories)
        dest_dir (str): Destination directory for balanced data
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of data for validation set
        random_state (int): Random seed for reproducibility
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    
    # Create destination directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (dest_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    class_dirs = [d for d in (src_dir / 'train').iterdir() if d.is_dir()]
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Create class directories in destination
        for split in splits:
            (dest_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Get all images for this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(class_dir.glob(f'**/{ext}'))
        
        if not image_files:
            print(f"  No images found in {class_dir}")
            continue
            
        print(f"  Found {len(image_files)} images")
        
        # Split into train and temp (val + test)
        train_files, temp_files = train_test_split(
            image_files, 
            test_size=(val_size + test_size), 
            random_state=random_state
        )
        
        # Split temp into val and test
        val_ratio = val_size / (val_size + test_size)
        val_files, test_files = train_test_split(
            temp_files, 
            test_size=(1 - val_ratio),
            random_state=random_state
        )
        
        print(f"  Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Copy files to destination
        for files, split in zip([train_files, val_files, test_files], splits):
            for src_file in tqdm(files, desc=f"  Copying to {split}", leave=False):
                dst_file = dest_dir / split / class_name / src_file.name
                if not dst_file.exists():  # Skip if already exists
                    shutil.copy2(src_file, dst_file)

def main():
    # Configuration
    src_data_dir = r"C:\tinkering\tinkering\data\processed_engagement_from_affectnet_folder"
    dest_data_dir = r"C:\tinkering\tinkering\data\balanced_engagement_data"
    
    print(f"Source directory: {src_data_dir}")
    print(f"Destination directory: {dest_data_dir}")
    
    # Balance the data
    balance_data_splits(
        src_dir=src_data_dir,
        dest_dir=dest_data_dir,
        test_size=0.15,  # 15% for test
        val_size=0.15,   # 15% for validation
        random_state=42  # For reproducibility
    )
    
    print("\nData balancing complete!")
    print(f"Balanced dataset saved to: {dest_data_dir}")

if __name__ == "__main__":
    main()
