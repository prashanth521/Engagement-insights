import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch

class DAiSEEDataset(Dataset):
    """DAiSEE dataset loader with 4-class engagement mapping.
    
    DAiSEE labels are continuous values [0,1] for each engagement dimension.
    We'll map these to 4 discrete classes based on the paper's methodology.
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the video frames.
            split (string): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Get list of samples
        self.samples = self._prepare_samples()
        
    def _load_annotations(self):
        """Load DAiSEE annotations."""
        annotations = []
        annotation_file = self.root_dir / 'annotations' / f'{self.split}.csv'
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
        df = pd.read_csv(annotation_file)
        
        # DAiSEE columns: frame_id, face_id, engagement, confidence, success, ...
        for _, row in df.iterrows():
            annotations.append({
                'frame_path': str(self.root_dir / 'frames' / row['frame_id']),
                'engagement': row['engagement'],
                'confidence': row['confidence']
            })
            
        return annotations
    
    def _prepare_samples(self):
        """Prepare samples by filtering valid frames and mapping to classes."""
        samples = []
        
        for ann in self.annotations:
            if not os.path.exists(ann['frame_path']):
                continue
                
            # Map continuous engagement score [0,1] to 4 classes
            engagement = ann['engagement']
            if engagement < 0.4:
                label = 0  # Disengaged
            elif engagement < 0.6:
                label = 1  # Barely-engaged
            elif engagement < 0.8:
                label = 2  # Engaged
            else:
                label = 3  # Highly-engaged
                
            samples.append({
                'image_path': ann['frame_path'],
                'label': label,
                'confidence': ann['confidence']
            })
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'confidence': sample['confidence']
        }

def download_daisee_dataset(output_dir='data/daisee'):
    """Download and extract DAiSEE dataset."""
    import requests
    import zipfile
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DAiSEE dataset URLs (you'll need to replace these with actual download links)
    urls = {
        'train': 'URL_TO_DAISEE_TRAIN_ZIP',
        'val': 'URL_TO_DAISEE_VAL_ZIP',
        'test': 'URL_TO_DAISEE_TEST_ZIP'
    }
    
    for split, url in urls.items():
        print(f"Downloading {split} split...")
        zip_path = output_dir / f'{split}.zip'
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {split}"):
                f.write(chunk)
                
        # Extract the file
        print(f"Extracting {split}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir / split)
            
        # Clean up zip file
        zip_path.unlink()
    
    print("DAiSEE dataset downloaded and extracted successfully!")

def create_annotations(data_dir='data/daisee'):
    """Create annotation CSVs for train/val/test splits."""
    data_dir = Path(data_dir)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"{split} directory not found, skipping...")
            continue
            
        annotations = []
        
        # Traverse through the directory structure
        for video_dir in tqdm(list(split_dir.glob('*/*')), desc=f"Processing {split}"):
            if not video_dir.is_dir():
                continue
                
            # Get all frames for this video
            frames = sorted(video_dir.glob('*.jpg'))
            
            # In a real scenario, you would load the actual engagement scores
            # For now, we'll use dummy values
            for frame_path in frames:
                annotations.append({
                    'frame_id': str(frame_path.relative_to(data_dir)),
                    'face_id': 0,  # Assuming one face per frame
                    'engagement': np.random.random(),  # Replace with actual engagement score
                    'confidence': 1.0,  # Replace with actual confidence
                    'success': 1  # 1 for success, 0 for failure
                })
        
        # Save annotations to CSV
        df = pd.DataFrame(annotations)
        (data_dir / 'annotations').mkdir(exist_ok=True)
        df.to_csv(data_dir / 'annotations' / f'{split}.csv', index=False)
        print(f"Saved {len(df)} annotations for {split} split")

def get_daisee_dataloaders(data_dir='data/daisee', batch_size=32, num_workers=4):
    """Get DataLoaders for DAiSEE dataset."""
    # Define transforms
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DAiSEEDataset(data_dir, split='train', transform=train_transform)
    val_dataset = DAiSEEDataset(data_dir, split='val', transform=val_transform)
    test_dataset = DAiSEEDataset(data_dir, split='test', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_names': ['disengaged', 'barely-engaged', 'engaged', 'highly-engaged']
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare DAiSEE dataset')
    parser.add_argument('--download', action='store_true', help='Download DAiSEE dataset')
    parser.add_argument('--data-dir', default='data/daisee', help='Directory to store dataset')
    args = parser.parse_args()
    
    if args.download:
        download_daisee_dataset(args.data_dir)
        create_annotations(args.data_dir)
