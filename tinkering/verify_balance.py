from pathlib import Path
import pandas as pd

def check_balance(data_dir):
    """Check the class distribution in the balanced dataset."""
    data_dir = Path(data_dir)
    
    results = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"{split} directory not found!")
            continue
            
        class_counts = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.*')))
                class_counts[class_dir.name] = count
        
        results[split] = class_counts
    
    # Create and display a nice table
    df = pd.DataFrame(results).fillna(0).astype(int)
    print("\nClass Distribution After Balancing:")
    print("-" * 50)
    print(df)
    
    # Print total counts
    print("\nTotal samples per split:")
    print(df.sum().to_string())
    
    return df

if __name__ == "__main__":
    balanced_dir = r"C:\tinkering\tinkering\data\balanced_engagement_data"
    check_balance(balanced_dir)
