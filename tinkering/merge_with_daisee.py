import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def map_daisee_to_engagement(daisee_engagement):
    """Map DAiSEE engagement scores to our class labels."""
    if daisee_engagement < 0.4:
        return 'disengaged'
    elif daisee_engagement < 0.6:
        return 'confused'
    elif daisee_engagement < 0.8:
        return 'attentive'
    else:
        return 'distracted'  # Highly engaged

def process_daisee_dataset(daisee_root, output_dir, samples_per_class=1000):
    """Process DAiSEE dataset and save in our format."""
    print("Processing DAiSEE dataset...")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ['attentive', 'distracted', 'confused', 'disengaged']:
            (output_dir / 'daisee' / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # In a real scenario, you would parse DAiSEE's annotations here
    # For now, we'll create a mock implementation
    print("Note: This is a placeholder. You'll need to implement DAiSEE data loading based on your DAiSEE dataset structure.")
    print("Please modify this function to load your DAiSEE data and call map_daisee_to_engagement() for each sample.")
    
    # Example structure (replace with actual DAiSEE loading code):
    """
    daisee_data = [
        # (image_path, engagement_score, split)
        ("path/to/daisee/image1.jpg", 0.75, 'train'),
        ("path/to/daisee/image2.jpg", 0.35, 'train'),
        # ...
    ]
    
    for img_path, engagement, split in tqdm(daisee_data, desc="Processing DAiSEE"):
        class_name = map_daisee_to_engagement(engagement)
        dest_path = output_dir / 'daisee' / split / class_name / Path(img_path).name
        shutil.copy2(img_path, dest_path)
    """

def merge_datasets(main_data_dir, daisee_data_dir, output_dir, samples_per_class=1000):
    """Merge main dataset with DAiSEE dataset."""
    print("Merging datasets...")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ['attentive', 'distracted', 'confused', 'disengaged']:
            (output_dir / 'merged' / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Function to copy files with progress
    def copy_files(src_dir, dest_dir, max_files=None):
        src_dir = Path(src_dir)
        if not src_dir.exists():
            return 0
            
        files = list(src_dir.glob('*.*'))
        if max_files and len(files) > max_files:
            files = random.sample(files, max_files)
            
        for src_file in tqdm(files, desc=f"Copying {src_dir.name}", leave=False):
            shutil.copy2(src_file, dest_dir / src_file.name)
        return len(files)
    
    # Copy main dataset
    print("\nProcessing main dataset...")
    for split in ['train', 'val', 'test']:
        for class_name in ['attentive', 'distracted', 'confused', 'disengaged']:
            src = main_data_dir / split / class_name
            dst = output_dir / 'merged' / split / class_name
            count = copy_files(src, dst)
            if count > 0:
                print(f"  Copied {count} files from {src}")
    
    # Copy DAiSEE dataset
    print("\nProcessing DAiSEE dataset...")
    for split in ['train', 'val', 'test']:
        for class_name in ['attentive', 'distracted', 'confused', 'disengaged']:
            src = daisee_data_dir / 'daisee' / split / class_name
            dst = output_dir / 'merged' / split / class_name
            count = copy_files(src, dst, samples_per_class // 3)  # Take 1/3 from DAiSEE
            if count > 0:
                print(f"  Copied {count} files from DAiSEE {split}/{class_name}")
    
    # Create final balanced splits
    print("\nCreating final balanced splits...")
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / 'merged' / split
        
        # Find minimum samples across classes in this split
        class_counts = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.*')))
                class_counts[class_dir.name] = count
        
        if not class_counts:
            continue
            
        min_count = min(class_counts.values())
        print(f"{split}: {min_count} samples per class")
        
        # Create balanced split
        balanced_dir = output_dir / 'final_balanced' / split
        balanced_dir.mkdir(parents=True, exist_ok=True)
        
        for class_name, count in class_counts.items():
            src_dir = split_dir / class_name
            dst_dir = balanced_dir / class_name
            dst_dir.mkdir(exist_ok=True)
            
            files = list(src_dir.glob('*.*'))
            if len(files) > min_count:
                files = random.sample(files, min_count)
                
            for src_file in tqdm(files, desc=f"Balancing {split}/{class_name}", leave=False):
                shutil.copy2(src_file, dst_dir / src_file.name)

def main():
    # Configuration
    base_dir = Path(r"C:\tinkering\tinkering\data")
    
    # Input directories
    main_data_dir = base_dir / "perfectly_balanced_data"
    daisee_data_dir = base_dir / "daisee_processed"
    
    # Output directory
    output_dir = base_dir / "merged_engagement_data"
    
    print(f"Main dataset: {main_data_dir}")
    print(f"DAiSEE dataset: {daisee_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process DAiSEE dataset (you'll need to implement the actual processing)
    # process_daisee_dataset(daisee_data_dir, output_dir)
    
    # Merge datasets
    merge_datasets(
        main_data_dir=main_data_dir,
        daisee_data_dir=daisee_data_dir,  # Will be used when DAiSEE processing is implemented
        output_dir=output_dir,
        samples_per_class=1000
    )
    
    print("\nDataset merging complete!")
    print(f"Merged dataset saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Implement DAiSEE data loading in 'process_daisee_dataset()'")
    print(f"2. Update 'daisee_data_dir' path to point to your DAiSEE dataset")
    print(f"3. Run this script again to create the merged dataset")
    print(f"4. Train your model using: python train_balanced.py --data-dir {output_dir / 'final_balanced'}")

if __name__ == "__main__":
    main()
